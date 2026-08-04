#include "physical/hbm/hbm_device.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
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

double align_to_validated_command_clock(
    double time_ns,
    double command_clock_period_ns) {
    require_nonnegative_timing(time_ns, "HBM command time");
    const double cycles = time_ns / command_clock_period_ns;
    if (!std::isfinite(cycles)) {
        throw std::runtime_error(
            "HBM command time exceeds the representable clock horizon");
    }
    // Reconstruct the lower grid point rather than applying an epsilon band.
    // A time produced by this same multiplication is stable, while a genuine
    // one-ULP-after-edge time advances to the next command cycle.
    double aligned_cycles = std::floor(cycles);
    double aligned_ns = aligned_cycles * command_clock_period_ns;
    if (aligned_ns >= time_ns) {
        require_nonnegative_timing(aligned_ns, "HBM aligned command time");
        return aligned_ns;
    }
    do {
        const double next_cycles = aligned_cycles + 1.0;
        if (next_cycles == aligned_cycles) {
            throw std::runtime_error(
                "HBM command time exceeds exact double-precision cycle indexing");
        }
        aligned_cycles = next_cycles;
        aligned_ns = aligned_cycles * command_clock_period_ns;
    } while (aligned_ns < time_ns);
    require_nonnegative_timing(aligned_ns, "HBM aligned command time");
    return aligned_ns;
}

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

void add_scaled_breakdown(
    Breakdown& target,
    const Breakdown& value,
    std::uint64_t multiplicity) {
    const auto factor = static_cast<double>(multiplicity);
    target.ingress_queue_wait_ns += value.ingress_queue_wait_ns * factor;
    target.scheduler_queue_wait_ns += value.scheduler_queue_wait_ns * factor;
    target.address_mapping_ns += value.address_mapping_ns * factor;
    target.translation_ns += value.translation_ns * factor;
    target.mapping_dram_ns += value.mapping_dram_ns * factor;
    target.refresh_stall_ns += value.refresh_stall_ns * factor;
    target.precharge_ns += value.precharge_ns * factor;
    target.activation_ns += value.activation_ns * factor;
    target.command_ns += value.command_ns * factor;
    target.array_read_ns += value.array_read_ns * factor;
    target.array_program_ns += value.array_program_ns * factor;
    target.array_erase_ns += value.array_erase_ns * factor;
    target.program_verify_ns += value.program_verify_ns * factor;
    target.media_lane_transfer_ns += value.media_lane_transfer_ns * factor;
    target.page_buffer_ns += value.page_buffer_ns * factor;
    target.sram_staging_ns += value.sram_staging_ns * factor;
    target.channel_transfer_ns += value.channel_transfer_ns * factor;
    target.tsv_transfer_ns += value.tsv_transfer_ns * factor;
    target.hb_io_transfer_ns += value.hb_io_transfer_ns * factor;
    target.transport_latency_ns += value.transport_latency_ns * factor;
    target.ecc_queue_wait_ns += value.ecc_queue_wait_ns * factor;
    target.ecc_latency_ns += value.ecc_latency_ns * factor;
    target.maintenance_ns += value.maintenance_ns * factor;
}

std::uint64_t modular_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    std::uint64_t modulus) {
    // Preconditions are enforced by the topology constructor/call sites.
    // Keeping both operands reduced avoids uint64 wrap changing the result
    // when modulus is not a power of two.
    return lhs >= modulus - rhs ? lhs - (modulus - rhs) : lhs + rhs;
}

std::uint64_t modular_subtract(
    std::uint64_t lhs,
    std::uint64_t rhs,
    std::uint64_t modulus) {
    return lhs >= rhs ? lhs - rhs : modulus - (rhs - lhs);
}

std::size_t command_index(HbmCommand command) {
    return static_cast<std::size_t>(command);
}

std::string command_name(HbmCommand command) {
    switch (command) {
    case HbmCommand::ACT:
        return "ACT";
    case HbmCommand::PRE:
        return "PRE";
    case HbmCommand::PREA:
        return "PREA";
    case HbmCommand::RD:
        return "RD";
    case HbmCommand::WR:
        return "WR";
    case HbmCommand::RDA:
        return "RDA";
    case HbmCommand::WRA:
        return "WRA";
    case HbmCommand::REFab:
        return "REFab";
    case HbmCommand::REFsb:
        return "REFsb";
    }
    throw std::runtime_error("unknown HBM command");
}

std::string level_name(HbmLevel level) {
    switch (level) {
    case HbmLevel::Channel:
        return "channel";
    case HbmLevel::PseudoChannel:
        return "pseudochannel";
    case HbmLevel::BankGroup:
        return "bankgroup";
    case HbmLevel::Bank:
        return "bank";
    case HbmLevel::Row:
        return "row";
    case HbmLevel::Column:
        return "column";
    }
    throw std::runtime_error("unknown HBM level");
}

HbmCommandMeta command_meta(HbmCommand command) {
    switch (command) {
    case HbmCommand::ACT:
        return {.opens_row = true};
    case HbmCommand::PRE:
    case HbmCommand::PREA:
        return {.closes_row = true};
    case HbmCommand::RD:
    case HbmCommand::WR:
        return {.accesses_column = true};
    case HbmCommand::RDA:
    case HbmCommand::WRA:
        return {.closes_row = true, .accesses_column = true};
    case HbmCommand::REFab:
    case HbmCommand::REFsb:
        return {.refreshes = true};
    }
    throw std::runtime_error("unknown HBM command");
}

HbmLevel command_scope(HbmCommand command) {
    switch (command) {
    case HbmCommand::ACT:
        return HbmLevel::Row;
    case HbmCommand::PRE:
    case HbmCommand::REFsb:
        return HbmLevel::Bank;
    case HbmCommand::PREA:
    case HbmCommand::REFab:
        return HbmLevel::Channel;
    case HbmCommand::RD:
    case HbmCommand::WR:
    case HbmCommand::RDA:
    case HbmCommand::WRA:
        return HbmLevel::Column;
    }
    throw std::runtime_error("unknown HBM command");
}

std::string timing_detail(
    HbmCommand command,
    HbmLevel level,
    std::string_view reason) {
    std::ostringstream out;
    out << command_name(command) << " waits for " << level_name(level);
    if (!reason.empty()) {
        out << " constraint " << reason;
    }
    return out.str();
}

void constrain_gate(
    HbmTimingGate& gate,
    double ready_ns,
    std::string_view reason) {
    if (ready_ns > gate.ready_ns) {
        gate.ready_ns = ready_ns;
        gate.reason = reason;
    }
}

} // namespace

std::uint64_t HbmConfig::pseudo_channel_width_bits() const {
    if (pseudo_channels_per_channel == 0 ||
        channel_width_bits % pseudo_channels_per_channel != 0) {
        throw std::runtime_error(
            "HBM channel width must divide evenly across pseudo-channels");
    }
    return channel_width_bits / pseudo_channels_per_channel;
}

std::uint64_t HbmConfig::row_size_bytes() const {
    if (pseudo_channels_per_channel == 0 ||
        channel_row_size_bytes % pseudo_channels_per_channel != 0) {
        throw std::runtime_error(
            "HBM channel row size must divide evenly across pseudo-channels");
    }
    return channel_row_size_bytes / pseudo_channels_per_channel;
}

std::uint64_t HbmConfig::burst_bytes() const {
    const auto pseudo_width = pseudo_channel_width_bits();
    if (pseudo_width % 8 != 0) {
        throw std::runtime_error("HBM pseudo-channel width must be byte-aligned");
    }
    return checked_mul(pseudo_width / 8, burst_length, "HBM burst bytes");
}

double HbmConfig::channel_bandwidth_GBps() const {
    return pin_rate_Gbps * static_cast<double>(channel_width_bits) / 8.0;
}

double HbmConfig::pseudo_channel_bandwidth_GBps() const {
    return pin_rate_Gbps * static_cast<double>(pseudo_channel_width_bits()) / 8.0;
}

double HbmConfig::command_clock_period_ns() const {
    return data_rate_per_command_clock / pin_rate_Gbps;
}

double HbmConfig::command_clock_MHz() const {
    return 1000.0 / command_clock_period_ns();
}

double HbmConfig::burst_duration_ns() const {
    return static_cast<double>(burst_length) / pin_rate_Gbps;
}

double HbmConfig::tCCD_S_ns() const {
    return tCCD_S_cycles * command_clock_period_ns();
}

double HbmConfig::tCCD_L_ns() const {
    return tCCD_L_cycles * command_clock_period_ns();
}

double HbmConfig::command_aligned_time_ns(double time_ns) const {
    const double tck_ns = command_clock_period_ns();
    require_positive_timing(tck_ns, "HBM derived command-clock period");
    return align_to_validated_command_clock(time_ns, tck_ns);
}

std::string HbmAddress::path() const {
    std::ostringstream out;
    out << "stack" << stack << "/ch" << channel << "/pch" << pseudo_channel
        << "/bg" << bank_group << "/bank" << bank << "/row" << row
        << "/off" << offset;
    return out.str();
}

double HbmStats::row_hit_rate() const {
    const auto total = row_hits + row_misses + row_conflicts;
    return total == 0 ? 0.0 : static_cast<double>(row_hits) / static_cast<double>(total);
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

HbmDevice::HbmDevice(HbmConfig config, AddressHeatmap* address_heatmap)
    : config_(config), address_heatmap_(address_heatmap) {
    require_positive_count(config_.capacity_bytes, "HBM capacity_bytes");
    require_positive_count(config_.stacks, "HBM stacks");
    require_positive_count(config_.channels_per_stack, "HBM channels_per_stack");
    require_positive_count(config_.pseudo_channels_per_channel, "HBM pseudo_channels_per_channel");
    require_positive_count(
        config_.bank_groups_per_pseudo_channel,
        "HBM bank_groups_per_pseudo_channel");
    require_positive_count(config_.banks_per_group, "HBM banks_per_group");
    require_positive_count(config_.channel_row_size_bytes, "HBM channel_row_size_bytes");
    require_positive_count(config_.channel_width_bits, "HBM channel_width_bits");
    require_positive_count(config_.burst_length, "HBM burst_length");
    require_positive_count(config_.queue_depth, "HBM queue_depth");
    require_positive_timing(config_.pin_rate_Gbps, "HBM pin_rate_Gbps");
    require_positive_count(
        config_.data_rate_per_command_clock,
        "HBM data_rate_per_command_clock");
    if (config_.channel_width_bits % 8 != 0) {
        throw std::runtime_error("HBM channel_width_bits must be byte-aligned");
    }
    const auto row_size_bytes = config_.row_size_bytes();
    const auto burst_bytes = config_.burst_bytes();
    if (burst_bytes > row_size_bytes || row_size_bytes % burst_bytes != 0) {
        throw std::runtime_error(
            "HBM derived burst bytes must divide the pseudo-channel row size");
    }
    if (config_.capacity_bytes % burst_bytes != 0) {
        throw std::runtime_error(
            "HBM capacity_bytes must be an integer number of physical bursts");
    }
    const double channel_bandwidth_GBps = config_.channel_bandwidth_GBps();
    const double pseudo_channel_bandwidth_GBps =
        config_.pseudo_channel_bandwidth_GBps();
    const double command_clock_period_ns = config_.command_clock_period_ns();
    const double command_clock_MHz = config_.command_clock_MHz();
    const double burst_duration_ns = config_.burst_duration_ns();
    const double tccd_s_ns = config_.tCCD_S_ns();
    const double tccd_l_ns = config_.tCCD_L_ns();
    require_positive_timing(channel_bandwidth_GBps, "HBM derived channel bandwidth");
    require_positive_timing(
        pseudo_channel_bandwidth_GBps,
        "HBM derived pseudo-channel bandwidth");
    require_positive_timing(command_clock_period_ns, "HBM derived command-clock period");
    require_positive_timing(command_clock_MHz, "HBM derived command-clock frequency");
    require_positive_timing(burst_duration_ns, "HBM derived burst duration");
    require_positive_timing(tccd_s_ns, "HBM derived tCCD_S");
    require_positive_timing(tccd_l_ns, "HBM derived tCCD_L");
    const double system_bandwidth_GBps = channel_bandwidth_GBps *
        static_cast<double>(config_.channels_per_stack) *
        static_cast<double>(config_.stacks);
    require_positive_timing(system_bandwidth_GBps, "HBM derived system bandwidth");
    if (config_.burst_length % config_.data_rate_per_command_clock != 0) {
        throw std::runtime_error(
            "HBM burst length must span a whole number of command-clock cycles");
    }
    const auto burst_command_cycles =
        config_.burst_length / config_.data_rate_per_command_clock;
    if (config_.tCCD_S_cycles < burst_command_cycles) {
        throw std::runtime_error(
            "HBM tCCD_S cycles cannot be shorter than one derived data burst");
    }
    if (config_.tCCD_L_cycles < config_.tCCD_S_cycles) {
        throw std::runtime_error("HBM tCCD_L cycles must be at least tCCD_S cycles");
    }
    require_nonnegative_timing(config_.address_mapping_ns, "HBM address_mapping_ns");
    require_positive_timing(config_.tRCDRD_ns, "HBM tRCDRD_ns");
    require_positive_timing(config_.tRCDWR_ns, "HBM tRCDWR_ns");
    require_positive_timing(config_.tCL_ns, "HBM tCL_ns");
    require_positive_timing(config_.tCWL_ns, "HBM tCWL_ns");
    require_positive_timing(config_.tRP_ns, "HBM tRP_ns");
    require_positive_timing(config_.tRAS_ns, "HBM tRAS_ns");
    require_positive_timing(config_.tRC_ns, "HBM tRC_ns");
    require_positive_timing(config_.tWR_ns, "HBM tWR_ns");
    require_positive_timing(config_.tRTP_ns, "HBM tRTP_ns");
    require_positive_count(config_.tCCD_S_cycles, "HBM tCCD_S_cycles");
    require_positive_count(config_.tCCD_L_cycles, "HBM tCCD_L_cycles");
    require_positive_timing(config_.tRRD_S_ns, "HBM tRRD_S_ns");
    require_positive_timing(config_.tRRD_L_ns, "HBM tRRD_L_ns");
    require_positive_timing(config_.tFAW_ns, "HBM tFAW_ns");
    require_positive_timing(config_.tWTR_S_ns, "HBM tWTR_S_ns");
    require_positive_timing(config_.tWTR_L_ns, "HBM tWTR_L_ns");
    require_positive_timing(config_.tRTW_ns, "HBM tRTW_ns");
    require_positive_timing(config_.tREFI_ns, "HBM tREFI_ns");
    require_positive_timing(config_.tRFC_ns, "HBM tRFC_ns");
    require_positive_timing(config_.tRFCsb_ns, "HBM tRFCsb_ns");
    require_positive_timing(config_.tRREFD_ns, "HBM tRREFD_ns");
    require_nonnegative_timing(config_.frfcfs_cap_ns, "HBM frfcfs_cap_ns");
    const double effective_trcdrd =
        config_.command_aligned_time_ns(config_.tRCDRD_ns);
    const double effective_trcdwr =
        config_.command_aligned_time_ns(config_.tRCDWR_ns);
    const double effective_tcl = config_.command_aligned_time_ns(config_.tCL_ns);
    const double effective_tcwl = config_.command_aligned_time_ns(config_.tCWL_ns);
    const double effective_trp = config_.command_aligned_time_ns(config_.tRP_ns);
    const double effective_tras = config_.command_aligned_time_ns(config_.tRAS_ns);
    const double effective_trc = config_.command_aligned_time_ns(config_.tRC_ns);
    const double effective_twr = config_.command_aligned_time_ns(config_.tWR_ns);
    const double effective_trtp = config_.command_aligned_time_ns(config_.tRTP_ns);
    const double effective_trrd_s =
        config_.command_aligned_time_ns(config_.tRRD_S_ns);
    const double effective_trrd_l =
        config_.command_aligned_time_ns(config_.tRRD_L_ns);
    const double effective_tfaw = config_.command_aligned_time_ns(config_.tFAW_ns);
    const double effective_twtr_s =
        config_.command_aligned_time_ns(config_.tWTR_S_ns);
    const double effective_twtr_l =
        config_.command_aligned_time_ns(config_.tWTR_L_ns);
    const double effective_trtw = config_.command_aligned_time_ns(config_.tRTW_ns);
    for (const auto effective : std::array<double, 15>{
             effective_trcdrd, effective_trcdwr, effective_tcl, effective_tcwl,
             effective_trp, effective_tras, effective_trc, effective_twr,
             effective_trtp, effective_trrd_s, effective_trrd_l, effective_tfaw,
             effective_twtr_s, effective_twtr_l, effective_trtw}) {
        require_positive_timing(effective, "HBM effective command timing");
    }
    require_positive_timing(
        effective_tras + effective_trp,
        "HBM effective tRAS+tRP timing");
    require_positive_timing(
        effective_tcl + burst_duration_ns,
        "HBM effective read command plus burst timing");
    require_positive_timing(
        effective_tcwl + burst_duration_ns + effective_twr,
        "HBM effective write recovery timing");
    require_positive_timing(
        effective_tcwl + burst_duration_ns + effective_twtr_s,
        "HBM effective short write-to-read timing");
    require_positive_timing(
        effective_tcwl + burst_duration_ns + effective_twtr_l,
        "HBM effective long write-to-read timing");
    if (config_.tRRD_L_ns < config_.tRRD_S_ns) {
        throw std::runtime_error("HBM tRRD_L must be at least tRRD_S");
    }
    if (config_.tWTR_L_ns < config_.tWTR_S_ns) {
        throw std::runtime_error("HBM tWTR_L must be at least tWTR_S");
    }
    if (config_.refresh_enabled) {
        const double refresh_duration =
            config_.same_bank_refresh ? config_.tRFCsb_ns : config_.tRFC_ns;
        if (refresh_duration + config_.tRREFD_ns >= config_.tREFI_ns) {
            throw std::runtime_error(
                "HBM refresh duration plus tRREFD must be smaller than tREFI");
        }
    }
    auto total_pseudo_channels = checked_mul(
        config_.stacks,
        config_.channels_per_stack,
        "HBM pseudo-channel topology");
    total_pseudo_channels = checked_mul(
        total_pseudo_channels,
        config_.pseudo_channels_per_channel,
        "HBM pseudo-channel topology");
    const auto banks_per_pseudo_channel = checked_mul(
        config_.bank_groups_per_pseudo_channel,
        config_.banks_per_group,
        "HBM bank topology");
    if (total_pseudo_channels > std::numeric_limits<std::size_t>::max() ||
        banks_per_pseudo_channel > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("HBM topology cannot be represented by size_t");
    }
    row_size_bytes_ = row_size_bytes;
    burst_bytes_ = burst_bytes;
    row_bursts_ = row_size_bytes / burst_bytes;
    total_pseudo_channels_ = total_pseudo_channels;
    banks_per_pseudo_channel_ = banks_per_pseudo_channel;
    pseudo_channels_per_stack_ = checked_mul(
        config_.channels_per_stack,
        config_.pseudo_channels_per_channel,
        "HBM pseudo-channels per stack");
    command_clock_period_ns_ = command_clock_period_ns;
    burst_duration_ns_ = burst_duration_ns;
    tccd_s_ns_ = tccd_s_ns;
    tccd_l_ns_ = tccd_l_ns;
    pseudo_channels_.resize(static_cast<std::size_t>(total_pseudo_channels));
    pseudo_channel_class_ids_.assign(pseudo_channels_.size(), 0);
    pseudo_channel_class_representatives_.assign(1, 0);
    pseudo_channel_class_state_keys_.assign(1, 0);
    route_ticket_markers_.assign(
        pseudo_channels_.size(),
        std::numeric_limits<std::uint64_t>::max());
    for (auto& pseudo_channel : pseudo_channels_) {
        pseudo_channel.banks.resize(static_cast<std::size_t>(banks_per_pseudo_channel));
        pseudo_channel.bank_groups.resize(config_.bank_groups_per_pseudo_channel);
        pseudo_channel.next_refresh_ns = config_.tREFI_ns;
    }
    refresh_parallel_stats();
}

namespace {

// Golden-ratio hash of the post-pseudo-channel stripe group, taking the HIGH
// multiplier bits: it
// decorrelates power-of-two request strides before pseudo-channel selection
// (low multiplier bits would still alias for even strides).
std::uint64_t pseudo_channel_hash(std::uint64_t group) {
    return (group * 0x9E3779B97F4A7C15ull) >> 32;
}

std::uint64_t bank_hash(std::uint64_t row_column) {
    return (row_column * 0xD1B54A32D192ED03ull) >> 32;
}

}  // namespace

HbmAddress HbmDevice::decode(std::uint64_t addr) const {
    if (addr >= config_.capacity_bytes) {
        throw std::runtime_error("HBM byte address is out of capacity");
    }
    const auto burst_linear = addr / burst_bytes_;
    const auto burst_offset = addr % burst_bytes_;
    const auto group = burst_linear / total_pseudo_channels_;
    // Burst-interleave stacks/channels/pseudo-channels first, then banks,
    // columns, and rows. This exposes the bank-group parallelism a real HBM
    // controller uses for sequential traffic instead of forcing every burst
    // of a row-sized host request through one tCCD_L chain. The reversible
    // swizzles avoid power-of-two stride aliases without changing the
    // reversible bank/row identity.
    const auto lane = burst_linear % total_pseudo_channels_;
    const auto swizzle =
        pseudo_channel_hash(group) % total_pseudo_channels_;
    const auto pseudo_channel_linear =
        modular_add(lane, swizzle, total_pseudo_channels_);
    const auto after_bank = group / banks_per_pseudo_channel_;
    const auto bank_lane = group % banks_per_pseudo_channel_;
    const auto bank_swizzle =
        bank_hash(after_bank) % banks_per_pseudo_channel_;
    const auto bank_interleave =
        modular_add(bank_lane, bank_swizzle, banks_per_pseudo_channel_);
    const auto column = after_bank % row_bursts_;
    HbmAddress decoded;
    decoded.stack = static_cast<std::uint32_t>(
        pseudo_channel_linear / pseudo_channels_per_stack_);
    const auto within_stack =
        pseudo_channel_linear % pseudo_channels_per_stack_;
    decoded.channel = static_cast<std::uint32_t>(
        within_stack / config_.pseudo_channels_per_channel);
    decoded.pseudo_channel = static_cast<std::uint32_t>(
        within_stack % config_.pseudo_channels_per_channel);
    // Bank-group bits are the lower bank-selection bits so sequential bursts
    // rotate groups before banks within a group and can use tCCD_S.
    decoded.bank_group = static_cast<std::uint32_t>(
        bank_interleave % config_.bank_groups_per_pseudo_channel);
    decoded.bank = static_cast<std::uint32_t>(
        bank_interleave / config_.bank_groups_per_pseudo_channel);
    decoded.row = after_bank / row_bursts_;
    decoded.offset = checked_add(
        checked_mul(column, burst_bytes_, "HBM decoded column offset"),
        burst_offset,
        "HBM decoded burst offset");
    return decoded;
}

std::uint64_t HbmDevice::encode(const HbmAddress& addr) const {
    if (addr.stack >= config_.stacks || addr.channel >= config_.channels_per_stack ||
        addr.pseudo_channel >= config_.pseudo_channels_per_channel ||
        addr.bank_group >= config_.bank_groups_per_pseudo_channel ||
        addr.bank >= config_.banks_per_group || addr.offset >= row_size_bytes_) {
        throw std::runtime_error("HBM address field out of range");
    }
    const auto pseudo_channel_linear =
        (static_cast<std::uint64_t>(addr.stack) * config_.channels_per_stack + addr.channel) *
        config_.pseudo_channels_per_channel + addr.pseudo_channel;
    const auto bank_interleave = static_cast<std::uint64_t>(addr.bank) *
        config_.bank_groups_per_pseudo_channel + addr.bank_group;
    const auto column = addr.offset / burst_bytes_;
    const auto burst_offset = addr.offset % burst_bytes_;
    const auto row_column = checked_add(
        checked_mul(addr.row, row_bursts_, "HBM encoded row columns"),
        column,
        "HBM encoded column");
    const auto bank_swizzle =
        bank_hash(row_column) % banks_per_pseudo_channel_;
    const auto bank_lane =
        modular_subtract(
            bank_interleave,
            bank_swizzle,
            banks_per_pseudo_channel_);
    const auto group = checked_add(
        checked_mul(
            row_column,
            banks_per_pseudo_channel_,
            "HBM encoded bank-interleaved group"),
        bank_lane,
        "HBM encoded bank");
    // Invert the pseudo-channel swizzle applied in decode().
    const auto swizzle =
        pseudo_channel_hash(group) % total_pseudo_channels_;
    const auto lane =
        modular_subtract(
            pseudo_channel_linear,
            swizzle,
            total_pseudo_channels_);
    const auto burst_linear = checked_add(
        checked_mul(group, total_pseudo_channels_, "HBM encoded line"),
        lane,
        "HBM encoded line");
    const auto encoded = checked_add(
        checked_mul(burst_linear, burst_bytes_, "HBM encoded byte address"),
        burst_offset,
        "HBM encoded byte address");
    if (encoded >= config_.capacity_bytes) {
        throw std::runtime_error("HBM encoded address is out of capacity");
    }
    return encoded;
}

PhysicalCompletion HbmDevice::issue(const PhysicalRequest& request) {
    if (auto isolated = try_issue_isolated_span(request)) {
        return std::move(*isolated);
    }
    return pump(enqueue(request));
}

void HbmDevice::validate_and_begin_request(const PhysicalRequest& request) {
    if (request.tier != Tier::HBM) {
        throw std::runtime_error("HbmDevice received non-HBM request");
    }
    if (request.op != Op::Read && request.op != Op::Write) {
        throw std::runtime_error("HbmDevice v0 supports read/write requests only");
    }
    if (request.bytes == 0) {
        throw std::runtime_error("HBM request bytes must be positive");
    }
    if (!std::isfinite(request.arrival_ns) || request.arrival_ns < 0.0) {
        throw std::runtime_error("HBM request arrival must be finite and non-negative");
    }
    if (request.addr >= config_.capacity_bytes ||
        request.bytes > config_.capacity_bytes - request.addr) {
        throw std::runtime_error("HBM request range is out of capacity");
    }
    if (last_enqueue_arrival_ns_ &&
        request.arrival_ns < *last_enqueue_arrival_ns_) {
        throw std::runtime_error(
            "HBM requests must be enqueued in nondecreasing arrival order; "
            "sort or explicitly admit the event stream before enqueueing");
    }
    last_enqueue_arrival_ns_ = request.arrival_ns;
    stats_.first_arrival_ns = std::min(stats_.first_arrival_ns, request.arrival_ns);
}

bool HbmDevice::same_pseudo_channel_state(
    const PseudoChannelState& lhs,
    const PseudoChannelState& rhs) const {
    const auto same_gate = [](const HbmTimingGate& left,
                              const HbmTimingGate& right) {
        return left.ready_ns == right.ready_ns &&
            left.reason.data() == right.reason.data() &&
            left.reason.size() == right.reason.size();
    };
    const auto same_gates = [&same_gate](const auto& left, const auto& right) {
        return std::equal(
            left.begin(), left.end(), right.begin(), right.end(), same_gate);
    };
    if (!lhs.queue.empty() || !rhs.queue.empty() ||
        lhs.banks.size() != rhs.banks.size() ||
        lhs.bank_groups.size() != rhs.bank_groups.size() ||
        lhs.bus_ready_ns != rhs.bus_ready_ns ||
        lhs.bus_busy_ns != rhs.bus_busy_ns ||
        lhs.next_refresh_ns != rhs.next_refresh_ns ||
        lhs.refresh_ready_ns != rhs.refresh_ready_ns ||
        lhs.refresh_epoch != rhs.refresh_epoch ||
        lhs.accesses != rhs.accesses ||
        lhs.recent_activations != rhs.recent_activations ||
        !same_gates(lhs.command_ready, rhs.command_ready)) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.banks.size(); ++index) {
        const auto& left = lhs.banks[index];
        const auto& right = rhs.banks[index];
        if (left.has_open_row != right.has_open_row ||
            left.open_row != right.open_row ||
            !same_gates(left.command_ready, right.command_ready)) {
            return false;
        }
    }
    for (std::size_t index = 0; index < lhs.bank_groups.size(); ++index) {
        if (!same_gates(
                lhs.bank_groups[index].command_ready,
                rhs.bank_groups[index].command_ready)) {
            return false;
        }
    }
    return true;
}

void HbmDevice::materialize_pseudo_channel_classes() {
    if (!pseudo_channel_classes_valid_) {
        return;
    }
    for (std::size_t index = 0; index < pseudo_channels_.size(); ++index) {
        const auto class_id = pseudo_channel_class_ids_.at(index);
        const auto representative =
            pseudo_channel_class_representatives_.at(class_id);
        if (index != representative) {
            pseudo_channels_[index] = pseudo_channels_[representative];
        }
    }
}

void HbmDevice::rebuild_pseudo_channel_classes() {
    if (pseudo_channel_classes_valid_) {
        return;
    }
    pseudo_channel_class_ids_.assign(pseudo_channels_.size(), 0);
    pseudo_channel_class_representatives_.clear();
    pseudo_channel_class_state_keys_.clear();
    previous_isolated_transitions_.clear();
    for (std::size_t index = 0; index < pseudo_channels_.size(); ++index) {
        bool matched = false;
        for (std::size_t class_id = 0;
             class_id < pseudo_channel_class_representatives_.size();
             ++class_id) {
            if (same_pseudo_channel_state(
                    pseudo_channels_[index],
                    pseudo_channels_[
                        pseudo_channel_class_representatives_[class_id]])) {
                pseudo_channel_class_ids_[index] =
                    static_cast<std::uint32_t>(class_id);
                matched = true;
                break;
            }
        }
        if (!matched) {
            if (pseudo_channel_class_representatives_.size() >=
                std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "HBM pseudo-channel class id space exhausted");
            }
            pseudo_channel_class_ids_[index] = static_cast<std::uint32_t>(
                pseudo_channel_class_representatives_.size());
            pseudo_channel_class_representatives_.push_back(index);
            if (next_pseudo_channel_state_key_ ==
                std::numeric_limits<std::uint64_t>::max()) {
                throw std::runtime_error(
                    "HBM pseudo-channel state key space exhausted");
            }
            pseudo_channel_class_state_keys_.push_back(
                next_pseudo_channel_state_key_++);
        }
    }
    pseudo_channel_classes_valid_ = true;
}

std::optional<PhysicalCompletion>
HbmDevice::try_issue_isolated_span(const PhysicalRequest& request) {
    // A diagnostics-free synchronous request starts with no queued controller
    // work. Every complete address-map stripe group presents the same local
    // bank/row/column sequence to every pseudo-channel; only the two partial
    // edge groups can select different channel subsets. Pseudo-channels in
    // the same exact controller-state class and edge signature therefore
    // execute identical events. Simulate one representative, multiply its
    // additive counters, and retain the class partition for the next span.
    // The ordinary enqueue/pump path remains the independent oracle whenever
    // diagnostics, unaligned bursts, or concurrent queued work are present.
    if (request.trace.retain_completion_diagnostics ||
        trace_spans_enabled(request.trace) ||
        !pending_.empty() ||
        std::any_of(
            pseudo_channels_.begin(),
            pseudo_channels_.end(),
            [](const PseudoChannelState& pseudo_channel) {
                return !pseudo_channel.queue.empty();
            }) ||
        request.addr % burst_bytes_ != 0 ||
        request.bytes % burst_bytes_ != 0) {
        return std::nullopt;
    }

    const auto burst_count = request.bytes / burst_bytes_;
    if (burst_count == 0) {
        return std::nullopt;
    }
    rebuild_pseudo_channel_classes();
    validate_and_begin_request(request);
    if (next_ticket_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error("HBM parent ticket space exhausted");
    }
    ++next_ticket_;

    if (address_heatmap_ != nullptr) {
        address_heatmap_->record_contiguous_accesses(
            AddressTrafficRecord{
                .domain = AddressDomain::HbmPhysical,
                .direction = request.op == Op::Read ?
                    TrafficDirection::Read : TrafficDirection::Write,
                .source = request.heatmap_source,
                .address = request.addr,
                .bytes = burst_bytes_,
            },
            burst_count);
    }

    const auto first_burst = request.addr / burst_bytes_;
    const auto end_burst = checked_add(
        first_burst, burst_count, "HBM isolated span burst end");
    const auto first_group = first_burst / total_pseudo_channels_;
    const auto last_group = (end_burst - 1) / total_pseudo_channels_;
    std::optional<std::uint64_t> first_edge_group;
    std::optional<std::uint64_t> last_edge_group;
    std::uint64_t full_group_begin = 0;
    std::uint64_t full_group_end = 0;
    std::vector<std::uint8_t> edge_signatures(
        pseudo_channels_.size(), 0);
    const auto mark_edge = [&](std::uint64_t group,
                               std::uint64_t lane_begin,
                               std::uint64_t lane_end,
                               std::uint8_t bit) {
        const auto swizzle =
            pseudo_channel_hash(group) % total_pseudo_channels_;
        for (auto lane = lane_begin; lane < lane_end; ++lane) {
            const auto pseudo_channel = modular_add(
                lane, swizzle, total_pseudo_channels_);
            edge_signatures.at(static_cast<std::size_t>(pseudo_channel)) |= bit;
        }
    };

    if (first_group == last_group) {
        first_edge_group = first_group;
        const auto lane_begin = first_burst % total_pseudo_channels_;
        auto lane_end = end_burst % total_pseudo_channels_;
        if (lane_end == 0) {
            lane_end = total_pseudo_channels_;
        }
        mark_edge(first_group, lane_begin, lane_end, 1);
    } else {
        full_group_begin = first_group;
        const auto first_lane = first_burst % total_pseudo_channels_;
        if (first_lane != 0) {
            first_edge_group = first_group;
            mark_edge(
                first_group,
                first_lane,
                total_pseudo_channels_,
                1);
            ++full_group_begin;
        }
        full_group_end = last_group + 1;
        const auto last_lane_end = end_burst % total_pseudo_channels_;
        if (last_lane_end != 0) {
            last_edge_group = last_group;
            mark_edge(last_group, 0, last_lane_end, 2);
            --full_group_end;
        }
    }

    using ClassKey = std::pair<std::uint32_t, std::uint8_t>;
    std::map<ClassKey, std::vector<std::size_t>> classes;
    for (std::size_t index = 0; index < pseudo_channels_.size(); ++index) {
        classes[{pseudo_channel_class_ids_.at(index),
                 edge_signatures[index]}].push_back(index);
    }
    std::vector<std::size_t> remaining_derived_classes(
        pseudo_channel_class_representatives_.size(), 0);
    for (const auto& [key, members] : classes) {
        (void)members;
        ++remaining_derived_classes.at(key.first);
    }
    // Move each old representative into a snapshot slot. A class that does
    // not split can then move straight through with no bank/gate allocation;
    // if it splits, only the non-final derived classes require a copy.
    std::vector<PseudoChannelState> old_states(
        pseudo_channel_class_representatives_.size());
    for (std::size_t class_id = 0;
         class_id < pseudo_channel_class_representatives_.size();
         ++class_id) {
        old_states[class_id] = std::move(pseudo_channels_.at(
            pseudo_channel_class_representatives_[class_id]));
    }

    PhysicalCompletion aggregate;
    aggregate.id = request.id;
    aggregate.tier = Tier::HBM;
    aggregate.op = request.op;
    aggregate.arrival_ns = request.arrival_ns;
    aggregate.start_ns = std::numeric_limits<double>::infinity();
    aggregate.logical_bytes = request.bytes;
    aggregate.physical_bytes = request.bytes;
    bool any_child = false;
    std::vector<std::uint32_t> next_class_ids(pseudo_channels_.size(), 0);
    std::vector<std::size_t> next_representatives;
    next_representatives.reserve(classes.size());
    std::vector<std::uint64_t> next_state_keys;
    next_state_keys.reserve(classes.size());
    std::vector<IsolatedTransition> current_transitions;
    current_transitions.reserve(classes.size());

    for (const auto& [key, members] : classes) {
        const auto signature = key.second;
        auto& remaining = remaining_derived_classes.at(key.first);
        if (remaining == 0) {
            throw std::runtime_error(
                "HBM isolated span lost a predecessor class");
        }
        --remaining;
        PseudoChannelState state = remaining == 0 ?
            std::move(old_states.at(key.first)) :
            old_states.at(key.first);
        const auto source_state_key =
            pseudo_channel_class_state_keys_.at(key.first);
        const auto representative = members.front();
        Breakdown representative_work;
        bool representative_has_child = false;
        PhysicalCompletion representative_completion;
        representative_completion.start_ns =
            std::numeric_limits<double>::infinity();
        std::uint64_t representative_children = 0;
        const auto before_read_bytes = stats_.read_bytes;
        const auto before_write_bytes = stats_.write_bytes;
        const auto before_row_hits = stats_.row_hits;
        const auto before_row_misses = stats_.row_misses;
        const auto before_row_conflicts = stats_.row_conflicts;
        const auto before_activations = stats_.activations;
        const auto before_precharges = stats_.precharges;
        const auto before_refresh_count = stats_.refresh_count;
        const auto before_bus_busy_ns = stats_.bus_busy_ns;

        const auto service_local = [&] {
            if (state.queue.empty()) {
                throw std::runtime_error(
                    "HBM isolated span serviced an empty local queue");
            }
            const auto picked = pick_next(state);
            for (std::size_t index = 0; index < picked; ++index) {
                if (state.queue[index].bypass_count >= config_.queue_depth) {
                    throw std::runtime_error(
                        "HBM isolated span exceeded bounded bypass");
                }
                ++state.queue[index].bypass_count;
            }
            auto entry = std::move(state.queue.at(picked));
            state.queue.erase(
                state.queue.begin() + static_cast<std::ptrdiff_t>(picked));
            auto child = service_request(state, entry.addr, entry.request);
            representative_work += child.breakdown;
            const bool critical = !representative_has_child ||
                child.finish_ns > representative_completion.finish_ns;
            representative_completion.start_ns = representative_has_child ?
                std::min(representative_completion.start_ns, child.start_ns) :
                child.start_ns;
            representative_completion.finish_ns = std::max(
                representative_completion.finish_ns, child.finish_ns);
            if (critical) {
                representative_completion.breakdown = child.breakdown;
            }
            representative_has_child = true;
        };
        const auto enqueue_group = [&](std::uint64_t group) {
            const auto swizzle =
                pseudo_channel_hash(group) % total_pseudo_channels_;
            const auto lane = modular_subtract(
                representative, swizzle, total_pseudo_channels_);
            const auto burst_linear = checked_add(
                checked_mul(
                    group,
                    total_pseudo_channels_,
                    "HBM isolated span group base"),
                lane,
                "HBM isolated span representative burst");
            const auto child_addr = checked_mul(
                burst_linear,
                burst_bytes_,
                "HBM isolated span representative address");
            const auto decoded = decode(child_addr);
            if (pseudo_channel_index(decoded) != representative) {
                throw std::runtime_error(
                    "HBM isolated span representative mapping drifted");
            }
            while (state.queue.size() >= config_.queue_depth) {
                service_local();
            }
            state.queue.push_back(QueuedRequest{
                .ticket = next_ticket_ - 1,
                .request = PhysicalRequest{
                    .tier = Tier::HBM,
                    .op = request.op,
                    .address_space = request.address_space,
                    .trace = request.trace,
                    .arrival_ns = request.arrival_ns,
                    .addr = child_addr,
                    .bytes = burst_bytes_,
                    .stream_id = request.stream_id,
                    .heatmap_source = request.heatmap_source,
                },
                .addr = decoded,
                .bypass_count = 0,
            });
            ++representative_children;
            stats_.max_queue_occupancy = std::max<std::uint64_t>(
                stats_.max_queue_occupancy, state.queue.size());
        };

        if (first_edge_group && (signature & 1) != 0) {
            enqueue_group(*first_edge_group);
        }
        for (auto group = full_group_begin;
             group < full_group_end;
             ++group) {
            enqueue_group(group);
        }
        if (last_edge_group && (signature & 2) != 0) {
            enqueue_group(*last_edge_group);
        }
        while (!state.queue.empty()) {
            service_local();
        }

        const auto multiplicity = static_cast<std::uint64_t>(members.size());
        const auto extra = multiplicity - 1;
        const auto replicate_counter = [&](std::uint64_t& value,
                                           std::uint64_t before,
                                           const char* name) {
            if (value < before) {
                throw std::runtime_error(
                    std::string("HBM isolated span counter regressed: ") +
                    name);
            }
            value = checked_add(
                value,
                checked_mul(value - before, extra, name),
                name);
        };
        replicate_counter(stats_.read_bytes, before_read_bytes,
                          "HBM isolated read bytes");
        replicate_counter(stats_.write_bytes, before_write_bytes,
                          "HBM isolated write bytes");
        replicate_counter(stats_.row_hits, before_row_hits,
                          "HBM isolated row hits");
        replicate_counter(stats_.row_misses, before_row_misses,
                          "HBM isolated row misses");
        replicate_counter(stats_.row_conflicts, before_row_conflicts,
                          "HBM isolated row conflicts");
        replicate_counter(stats_.activations, before_activations,
                          "HBM isolated activations");
        replicate_counter(stats_.precharges, before_precharges,
                          "HBM isolated precharges");
        replicate_counter(stats_.refresh_count, before_refresh_count,
                          "HBM isolated refreshes");
        // Derive both canonical transfer work and resource busy time from the
        // same representative sum. Subtracting a large pre-existing double
        // to recover the delta loses low bits and can make the two accounting
        // views disagree even though they describe identical burst work.
        stats_.bus_busy_ns = before_bus_busy_ns +
            representative_work.channel_transfer_ns *
            static_cast<double>(multiplicity);
        add_scaled_breakdown(
            stats_.stage_work, representative_work, multiplicity);

        const auto next_class_id = static_cast<std::uint32_t>(
            next_representatives.size());
        pseudo_channels_[representative] = std::move(state);
        next_representatives.push_back(representative);
        auto destination_state_key = source_state_key;
        if (representative_children != 0) {
            const IsolatedTransitionKey transition{
                .source_state_key = source_state_key,
                .arrival_bits = std::bit_cast<std::uint64_t>(
                    request.arrival_ns),
                .first_edge_group = first_edge_group.value_or(0),
                .full_group_begin = full_group_begin,
                .full_group_end = full_group_end,
                .last_edge_group = last_edge_group.value_or(0),
                .op = request.op,
                .edge_mask = signature,
            };
            const auto find_transition = [&](const auto& transitions) {
                return std::find_if(
                    transitions.begin(), transitions.end(),
                    [&](const IsolatedTransition& candidate) {
                        return candidate.transition == transition;
                    });
            };
            auto current = find_transition(current_transitions);
            if (current != current_transitions.end()) {
                destination_state_key = current->destination_state_key;
            } else {
                const auto previous =
                    find_transition(previous_isolated_transitions_);
                if (previous != previous_isolated_transitions_.end()) {
                    destination_state_key =
                        previous->destination_state_key;
                } else {
                    if (next_pseudo_channel_state_key_ ==
                        std::numeric_limits<std::uint64_t>::max()) {
                        throw std::runtime_error(
                            "HBM pseudo-channel state key space exhausted");
                    }
                    destination_state_key =
                        next_pseudo_channel_state_key_++;
                }
                current_transitions.push_back(IsolatedTransition{
                    .transition = transition,
                    .destination_state_key = destination_state_key,
                });
            }
        }
        next_state_keys.push_back(destination_state_key);
        for (const auto member : members) {
            next_class_ids[member] = next_class_id;
        }

        if (representative_has_child) {
            const bool critical = !any_child ||
                representative_completion.finish_ns > aggregate.finish_ns;
            aggregate.start_ns = any_child ?
                std::min(
                    aggregate.start_ns,
                    representative_completion.start_ns) :
                representative_completion.start_ns;
            aggregate.finish_ns = std::max(
                aggregate.finish_ns,
                representative_completion.finish_ns);
            if (critical) {
                aggregate.breakdown = representative_completion.breakdown;
            }
            any_child = true;
        } else if (representative_children != 0) {
            throw std::runtime_error(
                "HBM isolated span lost representative children");
        }
    }
    if (!any_child) {
        throw std::runtime_error("HBM isolated span performed no physical work");
    }
    // Different predecessor/signature classes can converge to the exact same
    // controller state (for example, the two 4 KiB halves of one 8 KiB stripe
    // after both have been accessed). A stable transition key identifies that
    // convergence without rescanning every bank and timing gate.
    std::vector<std::uint32_t> compact_class_ids(
        pseudo_channels_.size(), 0);
    std::vector<std::size_t> compact_representatives;
    std::vector<std::uint64_t> compact_state_keys;
    std::vector<std::uint32_t> old_to_compact(
        next_representatives.size(), 0);
    std::map<std::uint64_t, std::size_t> compact_by_state_key;
    for (std::size_t old_class_id = 0;
         old_class_id < next_representatives.size();
         ++old_class_id) {
        const auto representative = next_representatives[old_class_id];
        const auto state_key = next_state_keys.at(old_class_id);
        const auto found = compact_by_state_key.find(state_key);
        const auto compact_id = found == compact_by_state_key.end() ?
            compact_representatives.size() : found->second;
        if (found == compact_by_state_key.end()) {
            compact_by_state_key.emplace(state_key, compact_id);
            compact_representatives.push_back(representative);
            compact_state_keys.push_back(state_key);
        }
        if (compact_id >= std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "HBM pseudo-channel class id space exhausted");
        }
        old_to_compact[old_class_id] =
            static_cast<std::uint32_t>(compact_id);
    }
    for (std::size_t index = 0; index < next_class_ids.size(); ++index) {
        compact_class_ids[index] = old_to_compact.at(next_class_ids[index]);
    }
    pseudo_channel_class_ids_ = std::move(compact_class_ids);
    pseudo_channel_class_representatives_ =
        std::move(compact_representatives);
    pseudo_channel_class_state_keys_ = std::move(compact_state_keys);
    previous_isolated_transitions_ = std::move(current_transitions);
    pseudo_channel_classes_valid_ = true;
    return aggregate;
}

std::uint64_t HbmDevice::enqueue(const PhysicalRequest& request) {
    // The isolated-span path may retain only one materialized state per exact
    // pseudo-channel class. The general queued path can make every member
    // diverge independently, so expand the classes before touching a queue.
    materialize_pseudo_channel_classes();
    pseudo_channel_classes_valid_ = false;
    validate_and_begin_request(request);

    // A controller transaction may not cross a derived DRAM burst. Under the
    // burst-interleaved address map, each burst resolves to exactly one
    // pseudo-channel/bank/row/column tuple. Besides making every row decision
    // independently, this also
    // charges both bursts for an unaligned logical range that touches them.
    // Children are produced directly into their bounded controller queues:
    // retaining a complete fragment vector here used memory proportional to
    // the logical request *in addition to* the same children in the queues.
    if (next_ticket_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error("HBM parent ticket space exhausted");
    }
    const auto ticket = next_ticket_++;
    PhysicalCompletion aggregate;
    aggregate.id = request.id;
    aggregate.tier = Tier::HBM;
    aggregate.op = request.op;
    aggregate.arrival_ns = request.arrival_ns;
    aggregate.start_ns = std::numeric_limits<double>::infinity();
    aggregate.logical_bytes = request.bytes;
    pending_.emplace(ticket, PendingRequest{
        .completion = std::move(aggregate),
        .remaining_children = 0,
        .total_children = 0,
        .pseudo_channels = 0,
        .enqueue_complete = false,
        .has_child_completion = false,
        .retain_diagnostics = request.trace.retain_completion_diagnostics ||
            trace_spans_enabled(request.trace),
    });
    auto [route_it, route_inserted] =
        ticket_pseudo_channels_.emplace(ticket, std::vector<std::size_t>{});
    if (!route_inserted) {
        throw std::runtime_error("HBM generated a duplicate parent ticket");
    }
    auto& routes = route_it->second;

    const auto first_burst_base =
        request.addr - request.addr % burst_bytes_;
    const auto request_last = checked_add(
        request.addr,
        request.bytes - 1,
        "HBM request end address");
    const auto last_burst_base =
        request_last - request_last % burst_bytes_;
    const auto burst_count =
        (last_burst_base - first_burst_base) / burst_bytes_ + 1;
    routes.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(
        burst_count,
        pseudo_channels_.size())));
    if (address_heatmap_ != nullptr) {
        address_heatmap_->record_contiguous_accesses(
            AddressTrafficRecord{
                .domain = AddressDomain::HbmPhysical,
                .direction = request.op == Op::Read ?
                    TrafficDirection::Read : TrafficDirection::Write,
                .source = request.heatmap_source,
                .address = first_burst_base,
                .bytes = burst_bytes_,
            },
            burst_count);
    }
    std::uint64_t cursor = request.addr;
    std::uint64_t remaining = request.bytes;
    while (remaining != 0) {
        const auto burst_remaining =
            burst_bytes_ - cursor % burst_bytes_;
        const auto bytes = std::min(remaining, burst_remaining);
        PhysicalRequest child{
            .tier = request.tier,
            .op = request.op,
            .address_space = request.address_space,
            .trace = request.trace,
            .arrival_ns = request.arrival_ns,
            .addr = cursor,
            .bytes = bytes,
            .stream_id = request.stream_id,
            .heatmap_source = request.heatmap_source,
        };
        const auto addr = decode(cursor);
        const auto pc_index = pseudo_channel_index(addr);
        if (route_ticket_markers_.at(pc_index) != ticket) {
            route_ticket_markers_[pc_index] = ticket;
            routes.push_back(pc_index);
        }

        auto& queue = pseudo_channels_.at(pc_index).queue;
        // Admission is strictly bounded: service an existing entry before a
        // new child is inserted, never after temporarily exceeding the cap.
        while (queue.size() >= config_.queue_depth) {
            service_one(pc_index);
        }
        auto& parent = pending_.at(ticket);
        if (parent.remaining_children == std::numeric_limits<std::size_t>::max() ||
            parent.total_children == std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("HBM split request has too many children");
        }
        parent.remaining_children++;
        parent.total_children++;
        queue.push_back(QueuedRequest{
            .ticket = ticket,
            .request = std::move(child),
            .addr = addr,
            .bypass_count = 0,
        });
        stats_.max_queue_occupancy = std::max(
            stats_.max_queue_occupancy,
            static_cast<std::uint64_t>(queue.size()));

        remaining -= bytes;
        if (remaining != 0) {
            if (cursor > std::numeric_limits<std::uint64_t>::max() - bytes) {
                throw std::runtime_error("HBM request address range overflows uint64_t");
            }
            cursor += bytes;
        }
    }
    auto& parent = pending_.at(ticket);
    parent.pseudo_channels = routes.size();
    parent.enqueue_complete = true;
    return ticket;
}

PhysicalCompletion HbmDevice::pump(std::uint64_t ticket) {
    const auto routed = ticket_pseudo_channels_.find(ticket);
    if (routed == ticket_pseudo_channels_.end()) {
        throw std::runtime_error("HBM pump on unknown or already-pumped ticket");
    }
    while (completed_.find(ticket) == completed_.end()) {
        bool made_progress = false;
        for (const auto pc_index : routed->second) {
            if (!pseudo_channels_.at(pc_index).queue.empty()) {
                service_one(pc_index);
                made_progress = true;
            }
            if (completed_.find(ticket) != completed_.end()) {
                break;
            }
        }
        if (!made_progress) {
            throw std::runtime_error("HBM parent request lost an internal burst");
        }
    }
    auto completion = std::move(completed_.at(ticket));
    completed_.erase(ticket);
    ticket_pseudo_channels_.erase(routed);
    return completion;
}

void HbmDevice::drain_queues() {
    for (std::size_t i = 0; i < pseudo_channels_.size(); ++i) {
        while (!pseudo_channels_[i].queue.empty()) {
            service_one(i);
        }
    }
}

std::size_t HbmDevice::pick_next(const PseudoChannelState& pseudo_channel) const {
    // The queue is in arrival order, so the front is the oldest. FR-FCFS
    // picks the earliest ROW HIT among requests that are actually CONTENDING
    // with the oldest one. Both sides are previewed through address mapping,
    // command, data-bus, and refresh gates; looking only at arrival time lets
    // a not-yet-ready read hit delay a ready write conflict (and vice versa).
    // Time distance and a per-entry bypass counter jointly bound starvation.
    if (pseudo_channel.queue.size() == 1) {
        return 0;
    }
    const auto& oldest = pseudo_channel.queue.front();
    const auto oldest_preview = preview_first_command(pseudo_channel, oldest);
    for (std::size_t i = 0; i < pseudo_channel.queue.size(); ++i) {
        // Any candidate after a capped entry would bypass that entry once too
        // often. Tying the count cap to queue_depth gives each request at most
        // one full controller-window of younger priority, without introducing
        // a second externally configured queue-policy knob.
        if (i != 0 &&
            pseudo_channel.queue[i - 1].bypass_count >= config_.queue_depth) {
            break;
        }
        const auto& candidate = pseudo_channel.queue[i];
        if (candidate.request.arrival_ns - oldest.request.arrival_ns >
            config_.frfcfs_cap_ns) {
            break;
        }
        const auto candidate_preview =
            preview_first_command(pseudo_channel, candidate);
        if (candidate_preview.issue_ns > oldest_preview.issue_ns) {
            continue;
        }
        if (candidate_preview.row_hit) {
            return i;
        }
    }
    return 0;
}

HbmDevice::FirstCommandPreview HbmDevice::preview_first_command(
    const PseudoChannelState& pseudo_channel,
    const QueuedRequest& queued) const {
    const auto& bank = pseudo_channel.banks.at(bank_index(queued.addr));
    const auto& group = pseudo_channel.bank_groups.at(queued.addr.bank_group);
    const auto final = final_command(queued.request.op);
    bool has_open_row = bank.has_open_row;
    std::uint64_t open_row = bank.open_row;
    double next_refresh_ns = pseudo_channel.next_refresh_ns;
    double refresh_ready_ns = pseudo_channel.refresh_ready_ns;

    // Non-mutating projection of reserve_refresh_free_window(). It is
    // intentionally kept in lockstep with that scheduler path: an all-bank
    // refresh can turn a would-be PRE into ACT, or invalidate a row hit before
    // its RD/WR is actually issued.
    const auto project_refresh = [this, &next_refresh_ns, &refresh_ready_ns,
                                  &has_open_row](double& start_ns,
                                                 double duration_ns) {
        if (!config_.refresh_enabled) {
            return false;
        }
        const double trfc =
            config_.same_bank_refresh ? config_.tRFCsb_ns : config_.tRFC_ns;
        const double available_window_ns =
            config_.tREFI_ns - trfc - config_.tRREFD_ns;
        if (duration_ns > available_window_ns) {
            throw std::runtime_error(
                "HBM operation is longer than the refresh-free scheduling window");
        }
        bool closed_row = false;
        start_ns = align_command_time(std::max(start_ns, refresh_ready_ns));
        while (next_refresh_ns + trfc <= start_ns) {
            const double refresh_start = next_refresh_ns;
            next_refresh_ns += config_.tREFI_ns;
            refresh_ready_ns = std::max(
                refresh_ready_ns,
                refresh_start + trfc + config_.tRREFD_ns);
            start_ns = align_command_time(std::max(start_ns, refresh_ready_ns));
            if (!config_.same_bank_refresh) {
                has_open_row = false;
                closed_row = true;
            }
        }
        while (start_ns + duration_ns > next_refresh_ns) {
            const double blocked_until = next_refresh_ns + trfc;
            start_ns = align_command_time(std::max(
                start_ns,
                blocked_until + config_.tRREFD_ns));
            next_refresh_ns += config_.tREFI_ns;
            refresh_ready_ns = std::max(
                refresh_ready_ns,
                blocked_until + config_.tRREFD_ns);
            if (!config_.same_bank_refresh) {
                has_open_row = false;
                closed_row = true;
            }
        }
        return closed_row;
    };

    double ready_ns = queued.request.arrival_ns + config_.address_mapping_ns;
    (void)project_refresh(ready_ns, 0.0);
    for (;;) {
        HbmCommand command = final;
        if (!has_open_row) {
            command = HbmCommand::ACT;
        } else if (open_row != queued.addr.row) {
            command = HbmCommand::PRE;
        }
        const auto cmd_idx = command_index(command);
        double issue_ns = std::max({
            ready_ns,
            pseudo_channel.command_ready.at(cmd_idx).ready_ns,
            group.command_ready.at(cmd_idx).ready_ns,
            bank.command_ready.at(cmd_idx).ready_ns,
        });
        issue_ns = align_command_time(issue_ns);

        double duration_ns = 0.0;
        switch (command) {
        case HbmCommand::PRE:
            duration_ns = command_delay(config_.tRP_ns);
            break;
        case HbmCommand::ACT:
            duration_ns = final == HbmCommand::WR ?
                command_delay(config_.tRCDWR_ns) :
                command_delay(config_.tRCDRD_ns);
            break;
        case HbmCommand::RD:
        case HbmCommand::WR: {
            const double column_latency_ns = column_latency(command);
            issue_ns = std::max(
                issue_ns,
                pseudo_channel.bus_ready_ns - column_latency_ns);
            issue_ns = align_command_time(issue_ns);
            const double burst_ns = burst_duration_ns_;
            duration_ns = column_latency_ns + burst_ns;
            break;
        }
        default:
            throw std::runtime_error("HBM preview encountered an invalid access command");
        }

        const bool refresh_closed_row =
            project_refresh(issue_ns, duration_ns);
        if (refresh_closed_row &&
            (command == HbmCommand::PRE || command == HbmCommand::RD ||
             command == HbmCommand::WR)) {
            // The real scheduler returns issued=false in these cases and
            // retries from the newly closed-bank state.
            ready_ns = issue_ns;
            continue;
        }
        return FirstCommandPreview{
            .issue_ns = issue_ns,
            .row_hit = command == final,
        };
    }
}

void HbmDevice::service_one(std::size_t pseudo_channel_index) {
    auto& pseudo_channel = pseudo_channels_.at(pseudo_channel_index);
    if (pseudo_channel.queue.empty()) {
        throw std::runtime_error("HBM scheduler serviced an empty queue");
    }
    const auto picked = pick_next(pseudo_channel);
    for (std::size_t i = 0; i < picked; ++i) {
        if (pseudo_channel.queue[i].bypass_count >= config_.queue_depth) {
            throw std::runtime_error("HBM scheduler exceeded its bounded-bypass cap");
        }
        pseudo_channel.queue[i].bypass_count++;
    }
    QueuedRequest entry = std::move(pseudo_channel.queue.at(picked));
    pseudo_channel.queue.erase(pseudo_channel.queue.begin() +
        static_cast<std::ptrdiff_t>(picked));
    aggregate_child(
        entry.ticket,
        service_request(pseudo_channel, entry.addr, entry.request));
}

void HbmDevice::aggregate_child(std::uint64_t ticket, PhysicalCompletion child) {
    const auto found = pending_.find(ticket);
    if (found == pending_.end() || found->second.remaining_children == 0) {
        throw std::runtime_error("HBM completed an unknown internal burst");
    }
    auto& parent = found->second;
    auto& out = parent.completion;
    // Device totals represent physical work, so every burst child contributes.
    // The parent completion below retains only its latency-critical child's
    // compact diagnostic. Canonical additive work is accumulated separately
    // in stats_.stage_work so parallel children are neither lost nor confused
    // with elapsed wall time.
    stats_.stage_work += child.breakdown;
    const bool first = !parent.has_child_completion;
    parent.has_child_completion = true;
    const bool critical_child = first || child.finish_ns > out.finish_ns;
    if (first && parent.retain_diagnostics) {
        out.start_ns = child.start_ns;
        out.resource_path = child.resource_path;
        out.note = child.note;
    } else if (first) {
        out.start_ns = child.start_ns;
    } else {
        out.start_ns = std::min(out.start_ns, child.start_ns);
    }
    out.finish_ns = std::max(out.finish_ns, child.finish_ns);
    if (out.physical_bytes >
        std::numeric_limits<std::uint64_t>::max() - child.physical_bytes) {
        throw std::runtime_error("HBM aggregate physical byte count overflow");
    }
    out.physical_bytes += child.physical_bytes;
    // Retain the child on the parent's critical path for the compact
    // per-completion diagnostic. It is not the canonical parent work total:
    // physical work is stats_.stage_work and may legitimately exceed elapsed
    // latency when children overlap across pseudo-channels.
    if (critical_child) {
        out.breakdown = child.breakdown;
    }
    if (parent.retain_diagnostics) {
        out.spans.insert(out.spans.end(), child.spans.begin(), child.spans.end());
    }

    parent.remaining_children--;
    if (parent.remaining_children != 0 || !parent.enqueue_complete) {
        return;
    }
    if (parent.total_children > 1 && parent.retain_diagnostics) {
        out.resource_path = "hbm/split/" + std::to_string(parent.total_children) +
            "-bursts/" + std::to_string(parent.pseudo_channels) + "-pseudochannels";
        out.note = "split-burst-request";
    }
    completed_.emplace(ticket, std::move(out));
    pending_.erase(found);
}

PhysicalCompletion HbmDevice::service_request(
    PseudoChannelState& pseudo_channel,
    const HbmAddress& addr,
    const PhysicalRequest& request) {
    auto& bank = pseudo_channel.banks.at(bank_index(addr));
    const auto final = final_command(request.op);
    PhysicalCompletion out;
    out.id = request.id;
    out.tier = Tier::HBM;
    out.op = request.op;
    out.arrival_ns = request.arrival_ns;
    out.logical_bytes = request.bytes;
    out.physical_bytes = round_up(request.bytes, burst_bytes_);
    const bool retain_diagnostics =
        request.trace.retain_completion_diagnostics ||
        trace_spans_enabled(request.trace);
    if (retain_diagnostics) {
        out.resource_path = addr.path();
    }
    out.breakdown.address_mapping_ns = config_.address_mapping_ns;
    // Width, pin rate, pseudo-channel partitioning, and BL derive one exact
    // physical transfer. There is no independent nominal tBL to override it.
    const double burst_ns = burst_duration_ns_;
    auto* trace_spans = trace_spans_enabled(request.trace) ? &out.spans : nullptr;
    static const std::string no_trace_entity;
    const auto& trace_entity = retain_diagnostics ?
        out.resource_path : no_trace_entity;
    if (trace_spans != nullptr) {
        add_trace_span(
            trace_spans,
            "address_map",
            "mapping",
            trace_entity,
            request.arrival_ns,
            request.arrival_ns + out.breakdown.address_mapping_ns);
    }

    double ready_ns = request.arrival_ns + out.breakdown.address_mapping_ns;
    double first_command_ns = -1.0;
    const double clock_aligned_ready_ns = align_command_time(ready_ns);
    out.breakdown.scheduler_queue_wait_ns += clock_aligned_ready_ns - ready_ns;
    if (trace_spans != nullptr) {
        add_trace_span(
            trace_spans,
            "wait_command_clock",
            "queue",
            trace_entity,
            ready_ns,
            clock_aligned_ready_ns,
            true,
            "request becomes command-eligible on the next CK edge");
    }
    ready_ns = clock_aligned_ready_ns;
    out.breakdown.refresh_stall_ns += reserve_refresh_free_window(
        pseudo_channel,
        ready_ns,
        0.0,
        trace_spans,
        trace_entity);

    bool issued_precharge = false;
    bool issued_activation = false;
    for (;;) {
        if (bank.has_open_row && bank.open_row != addr.row) {
            const auto precharge = schedule_command(
                pseudo_channel,
                addr,
                HbmCommand::PRE,
                ready_ns,
                out.breakdown,
                trace_spans,
                trace_entity,
                first_command_ns,
                final);
            ready_ns = precharge.finish_ns;
            issued_precharge = issued_precharge || precharge.issued;
        }
        if (!bank.has_open_row) {
            const auto activation = schedule_command(
                pseudo_channel,
                addr,
                HbmCommand::ACT,
                ready_ns,
                out.breakdown,
                trace_spans,
                trace_entity,
                first_command_ns,
                final);
            ready_ns = activation.finish_ns;
            issued_activation = issued_activation || activation.issued;
        }

        // Make the CAS delay end when the data bus is available. A column
        // command must not issue and then have its data phase arbitrarily
        // stretched behind an unrelated burst.
        const double column_latency_ns = column_latency(final);
        const double bus_command_floor_ns =
            pseudo_channel.bus_ready_ns - column_latency_ns;
        if (bus_command_floor_ns > ready_ns) {
            out.breakdown.scheduler_queue_wait_ns += bus_command_floor_ns - ready_ns;
            if (trace_spans != nullptr) {
                add_trace_span(
                    trace_spans,
                    "wait_column_for_data_bus",
                    "queue",
                    trace_entity,
                    ready_ns,
                    bus_command_floor_ns,
                    true,
                    "delay RD/WR so CAS ends at data-bus availability");
            }
            ready_ns = bus_command_floor_ns;
        }
        const auto column = schedule_command(
            pseudo_channel,
            addr,
            final,
            ready_ns,
            out.breakdown,
            trace_spans,
            trace_entity,
            first_command_ns,
            final,
            burst_ns);
        ready_ns = column.finish_ns;
        if (column.issued) {
            break;
        }
        // An all-bank refresh landed while the column command was waiting and
        // closed the row. Reopen it before retrying the command.
    }

    if (issued_precharge) {
        stats_.row_conflicts++;
        if (retain_diagnostics) out.note = "row-conflict";
    } else if (issued_activation) {
        stats_.row_misses++;
        if (retain_diagnostics) out.note = "row-miss";
    } else {
        stats_.row_hits++;
        if (retain_diagnostics) out.note = "row-hit";
    }

    double bus_start = std::max(ready_ns, pseudo_channel.bus_ready_ns);
    out.breakdown.channel_transfer_ns = burst_ns;
    out.start_ns = first_command_ns >= 0.0 ? first_command_ns :
        request.arrival_ns + out.breakdown.address_mapping_ns;
    out.breakdown.scheduler_queue_wait_ns += std::max(0.0, bus_start - ready_ns);
    if (trace_spans != nullptr) {
        add_trace_span(
            trace_spans,
            "wait_data_bus",
            "queue",
            trace_entity,
            ready_ns,
            bus_start,
            true,
            "pseudochannel data bus busy");
    }
    const double finish = bus_start + burst_ns;
    out.finish_ns = finish;
    if (trace_spans != nullptr) {
        add_trace_span(
            trace_spans,
            request.op == Op::Read ? "read_burst" : "write_burst",
            "hbm_bus",
            "stack" + std::to_string(addr.stack) + "/ch" +
                std::to_string(addr.channel) + "/pch" +
                std::to_string(addr.pseudo_channel),
            bus_start,
            finish,
            true,
            std::to_string(out.physical_bytes) + "B");
    }

    pseudo_channel.bus_ready_ns = finish;
    pseudo_channel.bus_busy_ns += burst_ns;
    pseudo_channel.accesses++;

    if (request.op == Op::Read) {
        stats_.read_bytes += out.physical_bytes;
    } else {
        stats_.write_bytes += out.physical_bytes;
    }
    stats_.bus_busy_ns += out.breakdown.channel_transfer_ns;
    stats_.finish_ns = std::max(stats_.finish_ns, out.finish_ns);
    return out;
}

std::size_t HbmDevice::pseudo_channel_index(const HbmAddress& addr) const {
    return (static_cast<std::size_t>(addr.stack) * config_.channels_per_stack + addr.channel) *
        config_.pseudo_channels_per_channel + addr.pseudo_channel;
}

std::size_t HbmDevice::bank_index(const HbmAddress& addr) const {
    return static_cast<std::size_t>(addr.bank_group) * config_.banks_per_group + addr.bank;
}

HbmCommand HbmDevice::final_command(Op op) const {
    switch (op) {
    case Op::Read:
        return HbmCommand::RD;
    case Op::Write:
        return HbmCommand::WR;
    case Op::Erase:
    case Op::Refresh:
        break;
    }
    throw std::runtime_error("HBM supports read/write final commands only");
}

double HbmDevice::align_command_time(double time_ns) const {
    return align_to_validated_command_clock(
        time_ns, command_clock_period_ns_);
}

double HbmDevice::command_delay(double minimum_ns) const {
    return align_command_time(minimum_ns);
}

double HbmDevice::column_latency(HbmCommand command) const {
    switch (command) {
    case HbmCommand::RD:
        return command_delay(config_.tCL_ns);
    case HbmCommand::WR:
        return command_delay(config_.tCWL_ns);
    default:
        throw std::runtime_error("HBM column latency requested for a non-column command");
    }
}

HbmDevice::CommandSchedule HbmDevice::schedule_command(
    PseudoChannelState& pseudo_channel,
    HbmAddress addr,
    HbmCommand command,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& entity,
    double& first_command_ns,
    HbmCommand final_access,
    double data_burst_ns) {
    auto& bank = pseudo_channel.banks.at(bank_index(addr));
    auto& bank_group = pseudo_channel.bank_groups.at(addr.bank_group);
    const auto cmd_idx = command_index(command);

    struct Candidate {
        double ready_ns = 0.0;
        HbmLevel level = HbmLevel::Bank;
        std::string_view reason;
    };

    std::array<Candidate, 3> candidates{{
        Candidate{
            .ready_ns = pseudo_channel.command_ready.at(cmd_idx).ready_ns,
            .level = HbmLevel::PseudoChannel,
            .reason = pseudo_channel.command_ready.at(cmd_idx).reason,
        },
        Candidate{
            .ready_ns = bank_group.command_ready.at(cmd_idx).ready_ns,
            .level = HbmLevel::BankGroup,
            .reason = bank_group.command_ready.at(cmd_idx).reason,
        },
        Candidate{
            .ready_ns = bank.command_ready.at(cmd_idx).ready_ns,
            .level = HbmLevel::Bank,
            .reason = bank.command_ready.at(cmd_idx).reason,
        },
    }};

    double issue_ns = earliest_ns;
    HbmLevel blocking_level = HbmLevel::Bank;
    std::string_view blocking_reason;
    for (const auto& candidate : candidates) {
        if (candidate.ready_ns > issue_ns) {
            issue_ns = candidate.ready_ns;
            blocking_level = candidate.level;
            blocking_reason = candidate.reason;
        }
    }
    issue_ns = align_command_time(issue_ns);

    const auto wait_ns = std::max(0.0, issue_ns - earliest_ns);
    breakdown.scheduler_queue_wait_ns += wait_ns;
    if (spans != nullptr) {
        add_trace_span(
            spans,
            "wait_" + command_name(command),
            "queue",
            entity,
            earliest_ns,
            issue_ns,
            true,
            timing_detail(command, blocking_level, blocking_reason));
    }

    double duration_ns = 0.0;
    switch (command) {
    case HbmCommand::PRE:
        duration_ns = command_delay(config_.tRP_ns);
        break;
    case HbmCommand::ACT:
        duration_ns = final_access == HbmCommand::WR ?
            command_delay(config_.tRCDWR_ns) : command_delay(config_.tRCDRD_ns);
        break;
    case HbmCommand::RD:
        duration_ns = column_latency(HbmCommand::RD);
        break;
    case HbmCommand::WR:
        duration_ns = column_latency(HbmCommand::WR);
        break;
    default:
        // Refresh is modeled by reserve_refresh_free_window; auto-precharge and
        // refresh commands are never issued through the access scheduler.
        throw std::runtime_error("HBM command is not issued by the access scheduler");
    }

    // A column command, its CAS delay, and its complete data burst form one
    // refresh-free interval. Otherwise a refresh could incorrectly be placed
    // between RD/WR and the data that command launches.
    const auto refresh_epoch_before = pseudo_channel.refresh_epoch;
    const double refresh_reservation_ns = duration_ns +
        (command_meta(command).accesses_column ? data_burst_ns : 0.0);
    breakdown.refresh_stall_ns += reserve_refresh_free_window(
        pseudo_channel, issue_ns, refresh_reservation_ns, spans, entity);
    if (command == HbmCommand::PRE &&
        pseudo_channel.refresh_epoch != refresh_epoch_before &&
        !bank.has_open_row) {
        return CommandSchedule{
            .issue_ns = issue_ns,
            .finish_ns = issue_ns,
            .issued = false,
        };
    }
    if (command_meta(command).accesses_column &&
        pseudo_channel.refresh_epoch != refresh_epoch_before &&
        (!bank.has_open_row || bank.open_row != addr.row)) {
        return CommandSchedule{
            .issue_ns = issue_ns,
            .finish_ns = issue_ns,
            .issued = false,
        };
    }

    switch (command) {
    case HbmCommand::PRE:
        breakdown.precharge_ns += duration_ns;
        stats_.precharges++;
        break;
    case HbmCommand::ACT:
        breakdown.activation_ns += duration_ns;
        stats_.activations++;
        break;
    case HbmCommand::RD:
    case HbmCommand::WR:
        breakdown.command_ns += duration_ns;
        break;
    default:
        throw std::runtime_error("HBM command is not issued by the access scheduler");
    }
    if (first_command_ns < 0.0) {
        first_command_ns = issue_ns;
    }

    if (spans != nullptr) {
        std::ostringstream detail;
        const auto meta = command_meta(command);
        detail << "scope=" << level_name(command_scope(command));
        if (meta.opens_row) detail << " opens_row";
        if (meta.closes_row) detail << " closes_row";
        if (meta.accesses_column) detail << " accesses_column";
        if (meta.refreshes) detail << " refreshes";
        if (command == HbmCommand::ACT) {
            detail << " row=" << addr.row
                   << " next_" << command_name(final_access) << "_wait="
                   << (final_access == HbmCommand::WR ? "tRCDWR" : "tRCDRD");
        }
        add_trace_span(
            spans,
            command_name(command),
            command_meta(command).refreshes ? "refresh" : "hbm_command",
            entity,
            issue_ns,
            issue_ns + duration_ns,
            true,
            detail.str());
    }

    apply_command_state(
        pseudo_channel,
        addr,
        command,
        issue_ns,
        final_access,
        data_burst_ns);
    return CommandSchedule{
        .issue_ns = issue_ns,
        .finish_ns = issue_ns + duration_ns,
        .issued = true,
    };
}

double HbmDevice::reserve_refresh_free_window(
    PseudoChannelState& pseudo_channel,
    double& start_ns,
    double duration_ns,
    std::vector<TraceSpan>* spans,
    const std::string& entity) {
    if (!config_.refresh_enabled) {
        return 0.0;
    }
    const double trfc = config_.same_bank_refresh ? config_.tRFCsb_ns : config_.tRFC_ns;
    const double available_window_ns = config_.tREFI_ns - trfc - config_.tRREFD_ns;
    if (duration_ns > available_window_ns) {
        throw std::runtime_error(
            "HBM operation is longer than the refresh-free scheduling window");
    }

    const double original_start_ns = start_ns;
    start_ns = align_command_time(std::max(start_ns, pseudo_channel.refresh_ready_ns));

    // Account for every refresh whose complete blocking interval elapsed while
    // this pseudo-channel was idle. These refreshes still close all open rows;
    // skipping only their latency must not skip their architectural effect.
    while (pseudo_channel.next_refresh_ns + trfc <= start_ns) {
        const double refresh_start = pseudo_channel.next_refresh_ns;
        pseudo_channel.next_refresh_ns += config_.tREFI_ns;
        pseudo_channel.refresh_ready_ns = std::max(
            pseudo_channel.refresh_ready_ns,
            refresh_start + trfc + config_.tRREFD_ns);
        stats_.refresh_count++;
        apply_refresh_state(pseudo_channel);
        start_ns = align_command_time(std::max(start_ns, pseudo_channel.refresh_ready_ns));
    }

    // A shift past one refresh may overlap the next nominal window. Iterate
    // until the complete operation fits; realistic timing is guaranteed to
    // converge by the constructor's refresh-duty-cycle validation above.
    while (start_ns + duration_ns > pseudo_channel.next_refresh_ns) {
        const auto refresh_command = config_.same_bank_refresh ? HbmCommand::REFsb : HbmCommand::REFab;
        const double refresh_start = pseudo_channel.next_refresh_ns;
        const double blocked_until = refresh_start + trfc;
        add_trace_span(
            spans,
            command_name(refresh_command),
            "refresh",
            entity,
            refresh_start,
            blocked_until,
            true,
            config_.same_bank_refresh ? "same-bank refresh blocks data burst" :
                "all-bank refresh blocks data burst");
        start_ns = align_command_time(std::max(
            start_ns,
            blocked_until + config_.tRREFD_ns));
        // Advance by whole periods so the refresh cadence stays anchored to
        // tREFI instead of drifting later after every blocked access.
        pseudo_channel.next_refresh_ns += config_.tREFI_ns;
        pseudo_channel.refresh_ready_ns = std::max(
            pseudo_channel.refresh_ready_ns,
            blocked_until + config_.tRREFD_ns);
        stats_.refresh_count++;
        apply_refresh_state(pseudo_channel);
    }
    return std::max(0.0, start_ns - original_start_ns);
}

void HbmDevice::apply_refresh_state(PseudoChannelState& pseudo_channel) {
    if (config_.same_bank_refresh) {
        // REFsb bank rotation is not represented yet. Preserve the existing
        // conservative pseudo-channel stall without falsely closing every bank.
        return;
    }
    for (auto& bank : pseudo_channel.banks) {
        bank.has_open_row = false;
    }
    pseudo_channel.recent_activations.clear();
    pseudo_channel.refresh_epoch++;
}

void HbmDevice::apply_command_state(
    PseudoChannelState& pseudo_channel,
    const HbmAddress& addr,
    HbmCommand command,
    double issue_ns,
    HbmCommand final_access,
    double data_burst_ns) {
    auto& bank = pseudo_channel.banks.at(bank_index(addr));
    auto& bank_group = pseudo_channel.bank_groups.at(addr.bank_group);

    switch (command) {
    case HbmCommand::PRE:
        bank.has_open_row = false;
        constrain_gate(
            bank.command_ready.at(command_index(HbmCommand::ACT)),
            issue_ns + command_delay(config_.tRP_ns),
            "tRP (PRE->ACT)");
        break;
    case HbmCommand::ACT:
        bank.has_open_row = true;
        bank.open_row = addr.row;
        constrain_gate(
            bank.command_ready.at(command_index(HbmCommand::PRE)),
            issue_ns + command_delay(config_.tRAS_ns),
            "tRAS (ACT->PRE)");
        constrain_gate(
            bank.command_ready.at(command_index(HbmCommand::ACT)),
            issue_ns + std::max(
                command_delay(config_.tRC_ns),
                command_delay(config_.tRAS_ns) + command_delay(config_.tRP_ns)),
            "tRC (ACT->ACT)");
        constrain_gate(
            bank.command_ready.at(command_index(HbmCommand::RD)),
            issue_ns + command_delay(config_.tRCDRD_ns),
            "tRCDRD (ACT->RD)");
        constrain_gate(
            bank.command_ready.at(command_index(HbmCommand::WR)),
            issue_ns + command_delay(config_.tRCDWR_ns),
            "tRCDWR (ACT->WR)");
        constrain_gate(
            pseudo_channel.command_ready.at(command_index(HbmCommand::ACT)),
            issue_ns + command_delay(config_.tRRD_S_ns),
            "tRRD_S (ACT->ACT)");
        constrain_gate(
            bank_group.command_ready.at(command_index(HbmCommand::ACT)),
            issue_ns + command_delay(config_.tRRD_L_ns),
            "tRRD_L (ACT->ACT)");
        pseudo_channel.recent_activations.push_back(issue_ns);
        while (pseudo_channel.recent_activations.size() > 4) {
            pseudo_channel.recent_activations.pop_front();
        }
        // The next ACT is the 5th after the 4 retained here, so it must wait
        // for the oldest retained activation to leave the tFAW window.
        if (pseudo_channel.recent_activations.size() == 4) {
            constrain_gate(
                pseudo_channel.command_ready.at(command_index(HbmCommand::ACT)),
                pseudo_channel.recent_activations.front() +
                    command_delay(config_.tFAW_ns),
                "tFAW (ACT->ACT)");
        }
        break;
    case HbmCommand::RD:
        constrain_gate(
            bank.command_ready.at(command_index(HbmCommand::PRE)),
            issue_ns + command_delay(config_.tRTP_ns),
            "tRTP (RD->PRE)");
        constrain_gate(
            pseudo_channel.command_ready.at(command_index(HbmCommand::RD)),
            issue_ns + tccd_s_ns_,
            "tCCD_S (RD->RD)");
        constrain_gate(
            bank_group.command_ready.at(command_index(HbmCommand::RD)),
            issue_ns + tccd_l_ns_,
            "tCCD_L (RD->RD)");
        constrain_gate(
            pseudo_channel.command_ready.at(command_index(HbmCommand::WR)),
            issue_ns + command_delay(config_.tRTW_ns),
            "tRTW (RD->WR)");
        break;
    case HbmCommand::WR:
        constrain_gate(
            bank.command_ready.at(command_index(HbmCommand::PRE)),
            issue_ns + column_latency(HbmCommand::WR) + data_burst_ns +
                command_delay(config_.tWR_ns),
            "tCWL+data_burst+tWR (WR->PRE)");
        constrain_gate(
            pseudo_channel.command_ready.at(command_index(HbmCommand::WR)),
            issue_ns + tccd_s_ns_,
            "tCCD_S (WR->WR)");
        constrain_gate(
            bank_group.command_ready.at(command_index(HbmCommand::WR)),
            issue_ns + tccd_l_ns_,
            "tCCD_L (WR->WR)");
        constrain_gate(
            pseudo_channel.command_ready.at(command_index(HbmCommand::RD)),
            issue_ns + column_latency(HbmCommand::WR) + data_burst_ns +
                command_delay(config_.tWTR_S_ns),
            "tCWL+data_burst+tWTR_S (WR->RD)");
        constrain_gate(
            bank_group.command_ready.at(command_index(HbmCommand::RD)),
            issue_ns + column_latency(HbmCommand::WR) + data_burst_ns +
                command_delay(config_.tWTR_L_ns),
            "tCWL+data_burst+tWTR_L (WR->RD)");
        break;
    default:
        throw std::runtime_error("HBM command state update is not supported here");
    }
    (void)final_access;
}

void HbmDevice::refresh_parallel_stats() const {
    stats_.pseudo_channels = pseudo_channels_.size();
    stats_.active_pseudo_channels = 0;
    stats_.max_pseudo_channel_accesses = 0;
    stats_.max_pseudo_channel_busy_ns = 0.0;
    stats_.avg_active_pseudo_channel_busy_ns = 0.0;

    double active_busy_ns = 0.0;
    const auto accumulate = [&](const PseudoChannelState& pseudo_channel,
                                std::uint64_t multiplicity) {
        if (pseudo_channel.accesses == 0 && pseudo_channel.bus_busy_ns <= 0.0) {
            return;
        }
        stats_.active_pseudo_channels += multiplicity;
        active_busy_ns += pseudo_channel.bus_busy_ns *
            static_cast<double>(multiplicity);
        stats_.max_pseudo_channel_busy_ns =
            std::max(stats_.max_pseudo_channel_busy_ns, pseudo_channel.bus_busy_ns);
        stats_.max_pseudo_channel_accesses =
            std::max(stats_.max_pseudo_channel_accesses, pseudo_channel.accesses);
    };
    if (pseudo_channel_classes_valid_) {
        std::vector<std::uint64_t> populations(
            pseudo_channel_class_representatives_.size(), 0);
        for (const auto class_id : pseudo_channel_class_ids_) {
            if (class_id >= populations.size()) {
                throw std::runtime_error(
                    "HBM pseudo-channel class id is out of range");
            }
            ++populations[class_id];
        }
        for (std::size_t class_id = 0;
             class_id < pseudo_channel_class_representatives_.size();
             ++class_id) {
            accumulate(
                pseudo_channels_.at(
                    pseudo_channel_class_representatives_[class_id]),
                populations[class_id]);
        }
    } else {
        for (const auto& pseudo_channel : pseudo_channels_) {
            accumulate(pseudo_channel, 1);
        }
    }
    if (stats_.active_pseudo_channels != 0) {
        stats_.avg_active_pseudo_channel_busy_ns =
            active_busy_ns / static_cast<double>(stats_.active_pseudo_channels);
    }
    const auto work_matches = [](double lhs, double rhs) {
        const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
        return std::abs(lhs - rhs) <=
            32.0 * std::numeric_limits<double>::epsilon() * scale;
    };
    if (!work_matches(stats_.stage_work.channel_transfer_ns,
                      stats_.bus_busy_ns)) {
        std::ostringstream message;
        message << "HBM canonical stage work diverged from resource accounting: "
                << "stage_work.channel_transfer_ns="
                << stats_.stage_work.channel_transfer_ns
                << ", bus_busy_ns=" << stats_.bus_busy_ns;
        throw std::runtime_error(message.str());
    }
}

} // namespace hbfsim::physical::hbm
