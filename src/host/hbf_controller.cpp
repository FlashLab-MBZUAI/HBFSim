#include "host/hbf_controller.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <list>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace hbfsim::host {
using namespace hbfsim::physical;
namespace {

void require_positive_count(std::uint64_t value, const char* name) {
    if (value == 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
}

void require_positive_timing(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(std::string(name) + " must be positive and finite");
    }
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

std::uint64_t checked_ceil_div(
    std::uint64_t numerator,
    std::uint64_t denominator,
    const char* name) {
    if (denominator == 0) {
        throw std::runtime_error(std::string(name) + " has zero denominator");
    }
    return numerator == 0 ? 0 : 1 + (numerator - 1) / denominator;
}

// Stable controller-side address scrambler. AI tensor/KV layouts commonly
// advance by power-of-two page strides, so each stack-local mapping group
// rotates the global page lanes before ownership is encoded in the VPN.
// SplitMix64 is only a deterministic bit mixer; it models no randomness and
// has no mutable state.
std::uint64_t placement_mix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

using TemporalTouch = std::pair<double, std::uint64_t>;

std::optional<TemporalTouch> latest_touch_through(
    const std::set<TemporalTouch>& touches,
    double at_ns) {
    const auto after = touches.upper_bound(TemporalTouch{
        at_ns, std::numeric_limits<std::uint64_t>::max()});
    if (after == touches.begin()) {
        return std::nullopt;
    }
    return *std::prev(after);
}

std::string logic_entity(std::uint32_t stack) {
    return "stack" + std::to_string(stack) + "/logic";
}

std::string channel_entity(const HbfAddress& addr) {
    return "stack" + std::to_string(addr.stack) + "/ch" + std::to_string(addr.channel);
}

std::string die_entity(const HbfAddress& addr) {
    return channel_entity(addr) + "/die" + std::to_string(addr.die);
}

std::string plane_entity(const HbfAddress& addr) {
    return die_entity(addr) + "/plane" + std::to_string(addr.plane);
}

std::string subarray_entity(const HbfAddress& addr, std::size_t subarray) {
    return plane_entity(addr) + "/subarray" + std::to_string(subarray);
}

std::string media_lane_entity(const HbfAddress& addr, std::size_t lane) {
    return plane_entity(addr) + "/lane" + std::to_string(lane);
}

std::string page_buffer_bank_entity(const HbfAddress& addr, std::size_t bank) {
    return plane_entity(addr) + "/page_buffer_bank" + std::to_string(bank);
}

std::uint64_t metadata_lpn(std::uint64_t mapping_vpn) {
    return (std::uint64_t{1} << 63) | mapping_vpn;
}

bool is_metadata_lpn(std::uint64_t lpn) {
    return (lpn & (std::uint64_t{1} << 63)) != 0;
}

std::uint64_t metadata_vpn(std::uint64_t lpn) {
    return lpn & ~(std::uint64_t{1} << 63);
}

void trace_wait(
    std::vector<TraceSpan>* spans,
    const std::string& entity,
    double from_ns,
    double to_ns,
    const std::string& name = "scheduler_queue") {
    add_trace_span(spans, name, "queue", entity, from_ns, to_ns);
}

std::uint64_t range_overlap_bytes(
    std::uint64_t lhs_begin,
    std::uint64_t lhs_end,
    std::uint64_t rhs_begin,
    std::uint64_t rhs_end) {
    const auto begin = std::max(lhs_begin, rhs_begin);
    const auto end = std::min(lhs_end, rhs_end);
    return end > begin ? end - begin : 0;
}

std::uint64_t insert_merged_range(
    std::vector<DirtyRange>& ranges,
    DirtyRange incoming) {
    if (incoming.end <= incoming.begin) {
        return 0;
    }

    // Existing ranges are a sorted, disjoint last-writer map. Preserve the
    // source of bytes outside the incoming write, replace provenance only in
    // the overwritten interval, and coalesce adjacent ranges only when their
    // sources agree. This keeps dirty coverage and source attribution in one
    // canonical representation.
    std::uint64_t overlap = 0;
    std::vector<DirtyRange> merged;
    merged.reserve(ranges.size() + 2);
    const auto append = [&merged](DirtyRange range) {
        if (range.end <= range.begin) {
            return;
        }
        if (!merged.empty()) {
            auto& previous = merged.back();
            if (range.begin < previous.end) {
                throw std::runtime_error(
                    "HBF write-buffer provenance ranges overlap");
            }
            if (range.begin == previous.end &&
                range.heatmap_source == previous.heatmap_source) {
                previous.end = range.end;
                return;
            }
        }
        merged.push_back(range);
    };
    bool inserted = false;
    for (const auto& current : ranges) {
        if (current.end <= incoming.begin) {
            append(current);
            continue;
        }
        if (current.begin >= incoming.end) {
            if (!inserted) {
                append(incoming);
                inserted = true;
            }
            append(current);
            continue;
        }

        overlap += range_overlap_bytes(
            current.begin,
            current.end,
            incoming.begin,
            incoming.end);
        if (current.begin < incoming.begin) {
            append(DirtyRange{
                .begin = current.begin,
                .end = incoming.begin,
                .heatmap_source = current.heatmap_source,
            });
        }
        if (!inserted) {
            append(incoming);
            inserted = true;
        }
        if (current.end > incoming.end) {
            append(DirtyRange{
                .begin = incoming.end,
                .end = current.end,
                .heatmap_source = current.heatmap_source,
            });
        }
    }
    if (!inserted) {
        append(incoming);
    }
    ranges = std::move(merged);
    return overlap;
}

HeatmapTrafficSource dominant_dirty_source(
    const std::vector<DirtyRange>& ranges) {
    std::array<std::uint64_t, kHeatmapTrafficSourceCount> bytes_by_source{};
    for (const auto& range : ranges) {
        const auto source = static_cast<std::size_t>(range.heatmap_source);
        if (source >= bytes_by_source.size() || range.end <= range.begin) {
            throw std::runtime_error(
                "HBF write-buffer contains invalid source provenance");
        }
        const auto bytes = range.end - range.begin;
        if (bytes > std::numeric_limits<std::uint64_t>::max() -
                bytes_by_source[source]) {
            throw std::overflow_error(
                "HBF write-buffer source-byte accounting overflows uint64_t");
        }
        bytes_by_source[source] += bytes;
    }

    std::size_t dominant = bytes_by_source.size();
    std::uint64_t dominant_bytes = 0;
    for (std::size_t source = 0; source < bytes_by_source.size(); ++source) {
        // Iterating in enum order gives equal-byte ties a stable result.
        if (bytes_by_source[source] > dominant_bytes) {
            dominant = source;
            dominant_bytes = bytes_by_source[source];
        }
    }
    if (dominant == bytes_by_source.size()) {
        throw std::runtime_error(
            "HBF cannot flush a write-buffer entry without dirty provenance");
    }
    return static_cast<HeatmapTrafficSource>(dominant);
}

} // namespace

ResidentMappingCapacity derive_resident_mapping_capacity(
    const HbfConfig& config) {
    auto pages_per_stack = checked_mul(
        config.device.channels_per_stack,
        config.device.dies_per_channel,
        "HBF pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.device.planes_per_die,
        "HBF pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.device.blocks_per_plane,
        "HBF pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.device.pages_per_block,
        "HBF pages per stack");
    const auto mapping_pages_per_stack = checked_ceil_div(
        pages_per_stack,
        config.host.mapping_entries_per_page,
        "HBF resident mapping pages per stack");
    const auto bytes_per_stack = checked_mul(
        mapping_pages_per_stack,
        config.device.page_size_bytes,
        "HBF resident mapping bytes per stack");
    return ResidentMappingCapacity{
        .pages_per_stack = mapping_pages_per_stack,
        .bytes_per_stack = bytes_per_stack,
        .total_bytes = checked_mul(
            bytes_per_stack,
            config.device.stacks,
            "HBF resident mapping table bytes"),
    };
}

ControllerDramBudget derive_controller_dram_budget(
    const HbfConfig& config,
    std::uint64_t capacity_denominator) {
    if (capacity_denominator == 0) {
        throw std::runtime_error(
            "HBF controller-DRAM capacity denominator must be positive");
    }
    auto pages_per_stack = checked_mul(
        config.device.channels_per_stack,
        config.device.dies_per_channel,
        "HBF controller-DRAM raw pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.device.planes_per_die,
        "HBF controller-DRAM raw pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.device.blocks_per_plane,
        "HBF controller-DRAM raw pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.device.pages_per_block,
        "HBF controller-DRAM raw pages per stack");
    const auto budget_pages_per_stack =
        pages_per_stack / capacity_denominator;
    if (budget_pages_per_stack == 0) {
        throw std::runtime_error(
            "HBF controller-DRAM ratio yields less than one page per stack");
    }
    const auto bytes_per_stack = checked_mul(
        budget_pages_per_stack,
        config.device.page_size_bytes,
        "HBF controller-DRAM bytes per stack");
    return ControllerDramBudget{
        .bytes_per_stack = bytes_per_stack,
        .total_bytes = checked_mul(
            bytes_per_stack,
            config.device.stacks,
            "HBF controller-DRAM total bytes"),
    };
}

ControllerDramBudget derive_write_buffer_dram_capacity(
    const HbfConfig& config) {
    // With HBF device DRAM the write buffer lives on the device and takes no
    // HBM controller capacity.
    if (!config.host.write_coalescing_enabled ||
        config.host.device_dram_capacity_denominator != 0) {
        return ControllerDramBudget{};
    }
    const auto bytes_per_stack = checked_mul(
        config.host.write_buffer_pages,
        config.device.page_size_bytes,
        "HBF write-buffer controller-DRAM bytes per stack");
    return ControllerDramBudget{
        .bytes_per_stack = bytes_per_stack,
        .total_bytes = checked_mul(
            bytes_per_stack,
            config.device.stacks,
            "HBF write-buffer controller-DRAM total bytes"),
    };
}

DeviceDramCapacity derive_device_dram_capacity(const HbfConfig& config) {
    const auto denominator = config.host.device_dram_capacity_denominator;
    if (denominator == 0) {
        return DeviceDramCapacity{};
    }
    auto raw_pages_per_stack = checked_mul(
        config.device.channels_per_stack,
        config.device.dies_per_channel,
        "HBF device DRAM raw pages per stack");
    raw_pages_per_stack = checked_mul(
        raw_pages_per_stack,
        config.device.planes_per_die,
        "HBF device DRAM raw pages per stack");
    raw_pages_per_stack = checked_mul(
        raw_pages_per_stack,
        config.device.blocks_per_plane,
        "HBF device DRAM raw pages per stack");
    raw_pages_per_stack = checked_mul(
        raw_pages_per_stack,
        config.device.pages_per_block,
        "HBF device DRAM raw pages per stack");
    DeviceDramCapacity capacity;
    capacity.pages_per_stack = raw_pages_per_stack / denominator;
    capacity.write_buffer_pages_per_stack =
        config.host.write_coalescing_enabled ? config.host.write_buffer_pages : 0;
    if (capacity.pages_per_stack <= capacity.write_buffer_pages_per_stack) {
        throw std::runtime_error(
            "HBF device DRAM must hold the write buffer and at least one "
            "read-cache page per stack");
    }
    capacity.read_cache_pages_per_stack =
        capacity.pages_per_stack - capacity.write_buffer_pages_per_stack;
    capacity.bytes_per_stack = checked_mul(
        capacity.pages_per_stack,
        config.device.page_size_bytes,
        "HBF device DRAM bytes per stack");
    capacity.total_bytes = checked_mul(
        capacity.bytes_per_stack,
        config.device.stacks,
        "HBF device DRAM total bytes");
    return capacity;
}

const char* to_string(MappingMode mode) {
    switch (mode) {
    case MappingMode::FullResident:
        return "full-resident";
    case MappingMode::Cached:
        return "cached";
    case MappingMode::RawPhysical:
        return "raw-physical";
    }
    throw std::runtime_error("unknown HBF mapping mode");
}

MappingMode parse_mapping_mode(std::string_view value) {
    if (value == "full-resident") {
        return MappingMode::FullResident;
    }
    if (value == "cached") {
        return MappingMode::Cached;
    }
    if (value == "raw-physical") {
        return MappingMode::RawPhysical;
    }
    throw std::runtime_error(
        "HBF mapping mode must be full-resident, cached, or raw-physical");
}

const char* to_string(MappingCacheLayout layout) {
    switch (layout) {
    case MappingCacheLayout::Page: return "page";
    case MappingCacheLayout::Entry: return "entry";
    case MappingCacheLayout::Extent: return "extent";
    }
    throw std::runtime_error("unknown HBF mapping cache layout");
}

MappingCacheLayout parse_mapping_cache_layout(std::string_view value) {
    if (value == "page") return MappingCacheLayout::Page;
    if (value == "entry") return MappingCacheLayout::Entry;
    if (value == "extent") return MappingCacheLayout::Extent;
    throw std::runtime_error("HBF mapping cache layout must be page, entry, or extent");
}

const char* to_string(DeviceDramPolicy policy) {
    switch (policy) {
    case DeviceDramPolicy::Lru: return "lru";
    case DeviceDramPolicy::Fifo: return "fifo";
    }
    throw std::runtime_error("unknown HBF device DRAM policy");
}

DeviceDramPolicy parse_device_dram_policy(std::string_view value) {
    if (value == "lru") return DeviceDramPolicy::Lru;
    if (value == "fifo") return DeviceDramPolicy::Fifo;
    throw std::runtime_error("HBF device DRAM policy must be lru or fifo");
}

std::string HbfAddress::path() const {
    std::ostringstream out;
    out << "stack" << stack << "/ch" << channel << "/die" << die
        << "/plane" << plane << "/block" << block << "/page" << page
        << "/off" << offset;
    return out.str();
}

std::optional<double> HbfStats::waf() const {
    // Raw physical programs are host writes through the append-to-publish
    // interface: their payload is in the physical numerator, so it belongs
    // in the host-written denominator too. Counting it only above inflated
    // the WAF of any session that mixes published-extent appends with
    // page-mapped writes.
    const auto host_write_bytes = checked_add(
        logical_write_bytes,
        raw_physical_program_payload_bytes,
        "HBF WAF host-write denominator");
    if (host_write_bytes == 0) {
        return std::nullopt;
    }
    return static_cast<double>(physical_write_bytes) /
        static_cast<double>(host_write_bytes);
}

double HbfStats::active_span_ns() const {
    return std::isfinite(first_arrival_ns) && finish_ns > first_arrival_ns ?
        finish_ns - first_arrival_ns : 0.0;
}

double HbfStats::media_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || planes == 0 ? 0.0 :
        media_busy_ns / (span * static_cast<double>(planes));
}

double HbfStats::io_utilization() const {
    // One command port and independent read/write payload ports per stack.
    const auto span = active_span_ns();
    const auto resources = hbio_command_resources + hbio_data_resources;
    return span <= 0.0 || resources == 0 ? 0.0 :
        (hb_io_command_busy_ns + hb_io_data_busy_ns) /
            (span * static_cast<double>(resources));
}

double HbfStats::media_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : media_busy_ns / span;
}

double HbfStats::read_lane_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : read_lane_busy_ns / span;
}

double HbfStats::subarray_read_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : subarray_read_busy_ns / span;
}

double HbfStats::page_buffer_bank_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : page_buffer_bank_busy_ns / span;
}

double HbfStats::channel_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 :
        (channel_command_busy_ns + channel_data_busy_ns) / span;
}

double HbfStats::hbio_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 :
        (hb_io_command_busy_ns + hb_io_data_busy_ns) / span;
}

double HbfStats::hbio_command_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || hbio_command_resources == 0 ? 0.0 :
        hb_io_command_busy_ns / (span * static_cast<double>(hbio_command_resources));
}

double HbfStats::hbio_data_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || hbio_data_resources == 0 ? 0.0 :
        hb_io_data_busy_ns / (span * static_cast<double>(hbio_data_resources));
}

double HbfStats::sequencer_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : sequencer_busy_ns / span;
}

double HbfStats::ecc_issue_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : ecc_issue_busy_ns / span;
}

double HbfStats::ecc_issue_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || dies == 0 ? 0.0 :
        ecc_issue_busy_ns / (span * static_cast<double>(dies));
}


double HbfStats::plane_media_skew() const {
    return avg_active_plane_media_busy_ns <= 0.0 ? 0.0 :
        max_plane_media_busy_ns / avg_active_plane_media_busy_ns;
}

double HbfStats::plane_op_skew() const {
    return avg_active_plane_ops <= 0.0 ? 0.0 :
        static_cast<double>(max_plane_ops) / avg_active_plane_ops;
}

double HbfStats::media_lane_skew() const {
    return avg_active_media_lane_busy_ns <= 0.0 ? 0.0 :
        max_media_lane_busy_ns / avg_active_media_lane_busy_ns;
}

double HbfStats::media_lane_read_skew() const {
    return avg_active_media_lane_reads <= 0.0 ? 0.0 :
        static_cast<double>(max_media_lane_reads) / avg_active_media_lane_reads;
}

double HbfStats::subarray_busy_skew() const {
    return avg_active_subarray_busy_ns <= 0.0 ? 0.0 :
        max_subarray_busy_ns / avg_active_subarray_busy_ns;
}

double HbfStats::subarray_read_skew() const {
    return avg_active_subarray_reads <= 0.0 ? 0.0 :
        static_cast<double>(max_subarray_reads) / avg_active_subarray_reads;
}

double HbfStats::page_buffer_bank_skew() const {
    return avg_active_page_buffer_bank_busy_ns <= 0.0 ? 0.0 :
        max_page_buffer_bank_busy_ns / avg_active_page_buffer_bank_busy_ns;
}

double HbfStats::page_buffer_bank_read_skew() const {
    return avg_active_page_buffer_bank_reads <= 0.0 ? 0.0 :
        static_cast<double>(max_page_buffer_bank_reads) / avg_active_page_buffer_bank_reads;
}

double HbfStats::channel_busy_skew() const {
    return avg_active_channel_busy_ns <= 0.0 ? 0.0 :
        max_channel_busy_ns / avg_active_channel_busy_ns;
}

double HbfStats::die_transaction_skew() const {
    return avg_active_die_transactions <= 0.0 ? 0.0 :
        static_cast<double>(max_die_transactions) / avg_active_die_transactions;
}

HbfController::~HbfController() = default;
HbfController::HbfController(HbfController&&) noexcept = default;
HbfController& HbfController::operator=(HbfController&&) noexcept = default;

HbfController::HbfController(HbfConfig config, AddressHeatmap* address_heatmap)
    : config_(HbfMappingPolicy::media_config(config)), address_heatmap_(address_heatmap) {
    media_ = std::make_unique<physical::hbf::HbfDevice>(config_.device);
    require_positive_count(config_.device.stacks, "HBF stacks");
    require_positive_timing(config_.host.host_gc_decision_ns, "host GC decision time");
    require_positive_timing(config_.host.host_zone_remap_ns, "host zone remap time");
    require_positive_count(config_.host.zone_size_blocks, "HBF zone size in blocks");
    control_ready_ns_.assign(config_.device.stacks, 0.0);
    copy_ready_ns_.assign(config_.device.stacks, 0.0);
    require_positive_count(config_.device.channels_per_stack, "HBF channels_per_stack");
    require_positive_count(config_.device.dies_per_channel, "HBF dies_per_channel");
    require_positive_count(config_.device.planes_per_die, "HBF planes_per_die");
    require_positive_count(config_.device.blocks_per_plane, "HBF blocks_per_plane");
    require_positive_count(config_.device.pages_per_block, "HBF pages_per_block");
    require_positive_count(config_.device.page_size_bytes, "HBF page_size_bytes");
    require_positive_count(config_.device.media_lanes_per_plane, "HBF media_lanes_per_plane");
    require_positive_count(config_.device.page_buffer_banks_per_plane, "HBF page_buffer_banks_per_plane");

    if (config_.device.pages_per_block > 1024) {
        // BlockState's valid-page bitmap holds 16 x 64 bits.
        throw std::runtime_error("HBF pages_per_block must be <= 1024");
    }
    if (config_.device.page_size_bytes > 4096) {
        throw std::runtime_error("HBF v0 is SLC-only with page_size_bytes <= 4096");
    }
    require_positive_timing(config_.device.t_read_page_ns, "HBF t_read_page_ns");
    require_positive_timing(config_.device.t_program_page_ns, "HBF t_program_page_ns");
    require_positive_timing(config_.device.t_erase_block_ns, "HBF t_erase_block_ns");
    require_positive_timing(
        config_.device.ecc_decode_latency_ns, "HBF ecc_decode_latency_ns");
    require_positive_timing(
        config_.device.ecc_encode_latency_ns, "HBF ecc_encode_latency_ns");
    require_positive_timing(
        config_.device.ecc_decode_raw_bandwidth_GBps_per_die,
        "HBF ecc_decode_raw_bandwidth_GBps_per_die");
    require_positive_timing(
        config_.device.ecc_encode_raw_bandwidth_GBps_per_die,
        "HBF ecc_encode_raw_bandwidth_GBps_per_die");
    require_positive_timing(config_.device.channel_bandwidth_GBps, "HBF channel_bandwidth_GBps");
    require_positive_timing(config_.device.hb_io_bandwidth_GBps(), "HBF hb_io_bandwidth_GBps");
    require_positive_timing(config_.device.tsv_bandwidth_GBps, "HBF tsv_bandwidth_GBps");
    require_positive_timing(config_.device.media_lane_bandwidth_GBps, "HBF media_lane_bandwidth_GBps");
    require_positive_timing(config_.device.logic_sram_bandwidth_GBps, "HBF logic_sram_bandwidth_GBps");
    require_positive_timing(config_.device.page_buffer_bandwidth_GBps, "HBF page_buffer_bandwidth_GBps");
    require_positive_timing(config_.device.logic_scheduler_issue_ns, "HBF logic_scheduler_issue_ns");
    require_positive_timing(config_.device.address_generation_ns, "HBF address_generation_ns");
    require_positive_timing(config_.host.ctrl_dram_latency_ns, "HBF ctrl_dram_latency_ns");
    require_positive_timing(config_.host.ctrl_dram_issue_ns, "HBF ctrl_dram_issue_ns");
    require_positive_timing(config_.host.mapping_update_ns, "HBF mapping_update_ns");
    require_positive_timing(config_.host.local_mapping_bandwidth_GBps, "HBF local mapping bandwidth");
    require_positive_count(config_.host.local_mapping_burst_bytes, "HBF local mapping burst bytes");
    if (config_.device.page_size_bytes % config_.host.local_mapping_burst_bytes != 0)
        throw std::runtime_error("local mapping burst must divide an HBF page");
    require_positive_timing(config_.host.mapping_sram_latency_ns, "HBF mapping SRAM latency");
    require_positive_timing(config_.host.mapping_sram_issue_ns, "HBF mapping SRAM issue");
    if (config_.host.mapping_sram_bytes &&
        (config_.host.mapping_mode != MappingMode::FullResident || mapping_is_local() ||
         config_.host.mapping_sram_bytes % (mapping_sram_record_bytes * config_.device.stacks))) {
        throw std::runtime_error("mapping SRAM needs host full-resident mapping and whole 32-byte records per stack");
    }
    if (!std::isfinite(config_.host.mapping_control_compute_ns) || config_.host.mapping_control_compute_ns < 0 ||
        !std::isfinite(config_.host.mapping_internal_compute_ns) || config_.host.mapping_internal_compute_ns < 0)
        throw std::runtime_error("mapping compute costs must be finite and nonnegative");
    if (mapping_is_local() && (config_.host.auto_gc_enabled || config_.host.write_coalescing_enabled))
        throw std::runtime_error("device-local mapping is a no-GC, unbuffered architectural control");
    require_positive_timing(config_.host.free_page_allocation_ns, "HBF free_page_allocation_ns");
    require_positive_timing(config_.device.flash_tsu_issue_ns, "HBF flash_tsu_issue_ns");
    require_positive_count(config_.device.command_address_bytes, "HBF command_address_bytes");
    require_positive_count(config_.host.mapping_entries_per_page, "HBF mapping_entries_per_page");
    require_positive_count(
        config_.host.mapping_directory_entry_bytes,
        "HBF mapping_directory_entry_bytes");
    require_positive_count(
        config_.device.page_read_queue_depth_per_stack,
        "HBF page_read_queue_depth_per_stack");
    if (config_.device.page_read_queue_depth_per_stack >
        std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "HBF page_read_queue_depth_per_stack exceeds size_t range");
    }
    if (config_.device.oob_bytes_per_page >= config_.device.page_size_bytes) {
        throw std::runtime_error("HBF oob_bytes_per_page must be smaller than page_size_bytes");
    }
    const auto codeword_bytes = config_.device.page_size_bytes + config_.device.oob_bytes_per_page;
    const double decode_initiation_ns = transfer_time_ns(
        codeword_bytes, config_.device.ecc_decode_raw_bandwidth_GBps_per_die);
    const double encode_initiation_ns = transfer_time_ns(
        codeword_bytes, config_.device.ecc_encode_raw_bandwidth_GBps_per_die);
    if (config_.device.ecc_decode_latency_ns < decode_initiation_ns) {
        throw std::runtime_error(
            "HBF ecc_decode_latency_ns must cover the full codeword initiation interval");
    }
    if (config_.device.ecc_encode_latency_ns < encode_initiation_ns) {
        throw std::runtime_error(
            "HBF ecc_encode_latency_ns must cover the full codeword initiation interval");
    }
    if (config_.host.write_coalescing_enabled) {
        require_positive_count(config_.host.write_buffer_pages, "HBF write_buffer_pages");
    }
    if (config_.host.gc_wear_leveling_weight < 0.0 || !std::isfinite(config_.host.gc_wear_leveling_weight)) {
        throw std::runtime_error("HBF gc_wear_leveling_weight must be non-negative and finite");
    }
    if (config_.host.gc_relocation_pages_per_host_write == 0) {
        throw std::runtime_error(
            "HBF gc_relocation_pages_per_host_write must be positive");
    }
    if (config_.host.static_wear_leveling_erase_gap != 0 &&
        config_.host.static_wear_leveling_interval_erases == 0) {
        throw std::runtime_error(
            "HBF static_wear_leveling_interval_erases must be positive when "
            "static wear leveling is enabled");
    }
    if (!std::isfinite(config_.host.static_wear_leveling_max_write_fraction) ||
        config_.host.static_wear_leveling_max_write_fraction < 0.0 ||
        config_.host.static_wear_leveling_max_write_fraction > 1.0 ||
        config_.host.static_wear_leveling_stop_gap >
            config_.host.static_wear_leveling_erase_gap) {
        throw std::runtime_error("HBF invalid static wear-leveling budget or stop gap");
    }
    if (config_.device.thermal_start_at_ceiling && !config_.device.thermal_enabled) {
        throw std::runtime_error(
            "HBF thermal_start_at_ceiling requires thermal_enabled: a "
            "disabled thermal model has no governor to boot at its ceiling");
    }
    if (config_.device.thermal_enabled) {
        require_positive_timing(
            config_.device.thermal_resistance_c_per_w,
            "HBF thermal_resistance_c_per_w");
        require_positive_timing(
            config_.device.thermal_capacitance_j_per_c,
            "HBF thermal_capacitance_j_per_c");
        require_positive_timing(
            config_.device.thermal_read_energy_pj_per_bit,
            "HBF thermal_read_energy_pj_per_bit");
        require_positive_timing(
            config_.device.thermal_program_energy_pj_per_bit,
            "HBF thermal_program_energy_pj_per_bit");
        if (!std::isfinite(config_.device.thermal_ambient_c)) {
            throw std::runtime_error("HBF thermal_ambient_c must be finite");
        }
        if (!std::isfinite(config_.device.thermal_static_power_w) ||
            config_.device.thermal_static_power_w < 0.0) {
            throw std::runtime_error(
                "HBF thermal_static_power_w must be non-negative and finite");
        }
        if (!std::isfinite(config_.device.thermal_neighbor_heat_c) ||
            config_.device.thermal_neighbor_heat_c < 0.0) {
            throw std::runtime_error(
                "HBF thermal_neighbor_heat_c must be non-negative and finite");
        }
        if (!std::isfinite(config_.device.thermal_erase_energy_uj_per_block) ||
            config_.device.thermal_erase_energy_uj_per_block < 0.0) {
            throw std::runtime_error(
                "HBF thermal_erase_energy_uj_per_block must be non-negative "
                "and finite");
        }
        if (!std::isfinite(config_.device.thermal_throttle_c) ||
            !std::isfinite(config_.device.thermal_release_c) ||
            config_.device.thermal_release_c >= config_.device.thermal_throttle_c) {
            throw std::runtime_error(
                "HBF thermal_release_c must be finite and strictly below "
                "thermal_throttle_c");
        }
        const double boundary_c =
            config_.device.thermal_ambient_c + config_.device.thermal_neighbor_heat_c;
        thermal_idle_temperature_c_ = boundary_c +
            config_.device.thermal_static_power_w * config_.device.thermal_resistance_c_per_w;
        if (thermal_idle_temperature_c_ >= config_.device.thermal_release_c) {
            throw std::runtime_error(
                "HBF idle steady-state temperature (ambient + neighbor heat "
                "+ static power * resistance) must sit strictly below "
                "thermal_release_c: the governor could never release, and "
                "permanently throttled operation must be an explicit design "
                "decision, not a silent default (policy guard, not a "
                "physical operating limit)");
        }
        if (config_.device.thermal_throttle_power_w != 0.0 &&
            (!std::isfinite(config_.device.thermal_throttle_power_w) ||
             config_.device.thermal_throttle_power_w < 0.0)) {
            throw std::runtime_error(
                "HBF thermal_throttle_power_w must be non-negative and finite");
        }
        thermal_pacing_power_w_ = config_.device.thermal_throttle_power_w > 0.0 ?
            config_.device.thermal_throttle_power_w :
            (config_.device.thermal_throttle_c - boundary_c) /
                    config_.device.thermal_resistance_c_per_w -
                config_.device.thermal_static_power_w;
        if (!(thermal_pacing_power_w_ > 0.0)) {
            throw std::runtime_error(
                "HBF thermal pacing power resolved non-positive: raise "
                "thermal_throttle_c, lower the boundary (ambient + neighbor "
                "heat) or static power, or set thermal_throttle_power_w "
                "explicitly");
        }
        stats_.thermal_boundary_temperature_c = boundary_c;
        thermal_tau_ns_ = config_.device.thermal_resistance_c_per_w *
            config_.device.thermal_capacitance_j_per_c * 1e9;
        const auto payload_bits =
            static_cast<double>(config_.device.page_size_bytes) * 8.0;
        thermal_read_energy_j_ =
            config_.device.thermal_read_energy_pj_per_bit * payload_bits * 1e-12;
        thermal_program_energy_j_ =
            config_.device.thermal_program_energy_pj_per_bit * payload_bits * 1e-12;
        thermal_erase_energy_j_ =
            config_.device.thermal_erase_energy_uj_per_block * 1e-6;
        thermal_boot_temperature_c_ = config_.device.thermal_start_at_ceiling ?
            config_.device.thermal_throttle_c : thermal_idle_temperature_c_;
    }

    const auto total_channels = checked_mul(config_.device.stacks, config_.device.channels_per_stack,
        "HBF total_channels");
    const auto total_dies = checked_mul(total_channels, config_.device.dies_per_channel,
        "HBF total_dies");
    const auto total_planes = checked_mul(total_dies, config_.device.planes_per_die,
        "HBF total_planes");
    const auto total_blocks = checked_mul(total_planes, config_.device.blocks_per_plane,
        "HBF total_blocks");
    total_pages_ = checked_mul(total_blocks, config_.device.pages_per_block, "HBF total_pages");
    (void)checked_mul(total_pages_, config_.device.page_size_bytes, "HBF capacity bytes");
    if (config_.host.gc_reserved_free_blocks_per_plane >= config_.device.blocks_per_plane) {
        throw std::runtime_error(
            "HBF gc_reserved_free_blocks_per_plane must be smaller than blocks_per_plane");
    }
    const auto planes_per_stack = total_planes / config_.device.stacks;
    if (config_.host.auto_gc_enabled &&
        config_.host.mapping_mode != MappingMode::RawPhysical) {
        // The stack's GC-only reserve must hold one worst-case victim (and
        // the wear-leveling cold frontier); relocation draws on every plane
        // of the stack, so the requirement is per stack, not per plane.
        const auto reserve_pages = checked_mul(
            checked_mul(
                static_cast<std::uint64_t>(config_.host.gc_reserved_free_blocks_per_plane),
                planes_per_stack,
                "HBF GC reserve blocks per stack"),
            config_.device.pages_per_block,
            "HBF GC reserve pages per stack");
        const auto required = gc_reserve_requirement_pages();
        if (reserve_pages < required) {
            throw std::runtime_error(
                "HBF automatic GC requires a GC-only reserve of at least " +
                std::to_string(required) + " pages per stack "
                "(gc_reserved_free_blocks_per_plane x planes per stack x "
                "pages_per_block, currently " + std::to_string(reserve_pages) +
                "): one worst-case victim's live pages" +
                std::string(config_.host.mapping_mode == MappingMode::Cached ?
                    " plus the translation writebacks they induce" : "") +
                std::string(config_.host.static_wear_leveling_erase_gap != 0 ?
                    " plus one wear-leveling cold block" : ""));
        }
    }
    const auto usable_blocks_per_plane =
        config_.device.blocks_per_plane - config_.host.gc_reserved_free_blocks_per_plane;
    const auto usable_blocks_per_stack = checked_mul(
        planes_per_stack, usable_blocks_per_plane, "HBF usable blocks per stack");
    const auto gc_watermark_capacity = checked_mul(
        usable_blocks_per_stack, config_.device.pages_per_block, "HBF usable pages per stack");
    if (config_.host.gc_low_watermark_pages > gc_watermark_capacity) {
        throw std::runtime_error(
            "HBF gc_low_watermark_pages exceeds usable per-stack page capacity");
    }
    if (config_.host.gc_hard_watermark_pages > gc_watermark_capacity) {
        throw std::runtime_error(
            "HBF gc_hard_watermark_pages exceeds usable per-stack page capacity");
    }
    free_pages_ = total_pages_;
    free_pages_per_stack_.assign(config_.device.stacks, total_pages_ / config_.device.stacks);
    next_data_allocation_plane_per_stack_.assign(config_.device.stacks, 0);
    next_mapping_allocation_plane_per_stack_.assign(config_.device.stacks, 0);
    next_gc_allocation_plane_per_stack_.assign(config_.device.stacks, 0);
    causal_state_ready_by_stack_.assign(config_.device.stacks, 0.0);
    state_observation_by_stack_.assign(config_.device.stacks, 0.0);
    gc_active_by_stack_.assign(config_.device.stacks, false);
    gc_victim_by_stack_.assign(config_.device.stacks, std::nullopt);
    wear_leveling_by_stack_.assign(config_.device.stacks, std::nullopt);
    pending_wear_check_by_stack_.assign(config_.device.stacks, std::nullopt);
    wear_leveling_engaged_by_stack_.assign(config_.device.stacks, false);
    wear_leveling_credit_by_stack_.assign(config_.device.stacks, 0.0);
    wear_leveling_last_move_by_block_.resize(static_cast<std::size_t>(total_blocks));
    wear_leveling_source_runs_by_block_.assign(static_cast<std::size_t>(total_blocks), 0);
    cold_block_by_stack_.assign(config_.device.stacks, std::nullopt);
    dirty_mapping_pages_by_stack_.assign(config_.device.stacks, 0);
    pending_commit_keys_by_stack_.assign(config_.device.stacks, {});
    materialized_ready_by_block_.assign(
        static_cast<std::size_t>(total_blocks), 0.0);
    write_buffer_lru_by_stack_.resize(config_.device.stacks);
    write_buffer_by_stack_.resize(config_.device.stacks);
    write_buffer_slot_release_by_stack_.resize(config_.device.stacks);
    if (config_.host.device_dram_capacity_denominator != 0) {
        if (!std::isfinite(config_.host.device_dram_hit_latency_ns) ||
            config_.host.device_dram_hit_latency_ns < 0.0 ||
            !std::isfinite(config_.host.device_dram_bandwidth_GBps) ||
            config_.host.device_dram_bandwidth_GBps <= 0.0) {
            throw std::runtime_error(
                "HBF device DRAM needs a finite non-negative hit latency and positive bandwidth");
        }
        // The requested config carries the structural policy's write buffer
        // (the media config clears it), so both organizations carve it here.
        device_dram_ = derive_device_dram_capacity(config);
        device_dram_order_by_stack_.resize(config_.device.stacks);
        device_dram_by_stack_.resize(config_.device.stacks);
        device_dram_port_by_stack_.resize(config_.device.stacks);
        device_dram_fill_by_stack_.resize(config_.device.stacks);
        stats_.device_dram_capacity_bytes = device_dram_.total_bytes;
        stats_.device_dram_capacity_bytes_per_stack = device_dram_.bytes_per_stack;
        stats_.device_dram_read_cache_pages_per_stack = device_dram_.read_cache_pages_per_stack;
    }
    page_read_credit_release_by_stack_.resize(config_.device.stacks);
    for (auto& releases : page_read_credit_release_by_stack_)
        releases.reserve(config_.device.page_read_queue_depth_per_stack);
    mapping_cache_lru_by_stack_.resize(config_.device.stacks);
    mapping_cache_by_stack_.resize(config_.device.stacks);
    mapping_cache_bytes_by_stack_.resize(config_.device.stacks, 0);
    mapping_scratch_by_stack_.resize(config_.device.stacks);
    for (auto& pages : mapping_scratch_by_stack_)
        pages.resize(config_.host.mapping_scratch_pages);
    if (config_.host.mapping_cache_layout != MappingCacheLayout::Page &&
        (config_.host.mapping_mode != MappingMode::Cached ||
         config_.host.mapping_cache_tag_bytes < 16 || config_.host.mapping_scratch_pages == 0)) {
        throw std::runtime_error("entry/extent mapping needs cached mode, tags and a bounded mapping scratch page");
    }
    if (!std::isfinite(config_.host.mapping_codec_ns_per_entry) ||
        config_.host.mapping_codec_ns_per_entry < 0.0) {
        throw std::runtime_error("invalid mapping codec service time");
    }
    if (config_.device.thermal_enabled) {
        thermal_stacks_.resize(config_.device.stacks);
        for (auto& stack : thermal_stacks_) {
            stack.node.temperature_c = thermal_boot_temperature_c_;
            stack.node.peak_c = thermal_boot_temperature_c_;
            stack.node.throttled = config_.device.thermal_start_at_ceiling;
        }
        stats_.thermal_enabled = true;
        stats_.thermal_boot_temperature_c = thermal_boot_temperature_c_;
    }

    // The logical page table has the same entry density as its persistent
    // checkpoint pages. FullResident provisions the complete footprint;
    // Cached provisions only an explicit page-cache budget.
    if (config_.host.ctrl_dram_capacity_denominator != 0) {
        const auto derived = derive_controller_dram_budget(
            config_, config_.host.ctrl_dram_capacity_denominator);
        if (config_.host.ctrl_dram_bytes == 0) {
            config_.host.ctrl_dram_bytes = derived.total_bytes;
        } else if (config_.host.ctrl_dram_bytes != derived.total_bytes) {
            throw std::runtime_error(
                "HBF resolved ctrl_dram_bytes differs from its physical-"
                "capacity denominator");
        }
    }
    const auto resident_mapping =
        derive_resident_mapping_capacity(config_);
    mapping_table_pages_per_stack_ =
        resident_mapping.pages_per_stack;
    mapping_table_bytes_per_stack_ =
        resident_mapping.bytes_per_stack;
    const auto required_ctrl_dram_bytes =
        resident_mapping.total_bytes;
    const auto write_buffer_dram =
        derive_write_buffer_dram_capacity(config_);
    const auto copy_per_stack = config_.host.auto_gc_enabled &&
        config_.host.mapping_mode != MappingMode::RawPhysical ? config_.device.page_size_bytes : 0;
    stats_.host_gc_buffer_bytes = copy_per_stack * config_.device.stacks;
    const auto scratch_per_stack = checked_mul(config_.host.mapping_scratch_pages,
        checked_add(config_.device.page_size_bytes, config_.host.mapping_cache_tag_bytes,
            "tagged mapping scratch page"), "mapping scratch bytes per stack");
    stats_.mapping_scratch_capacity_bytes = checked_mul(scratch_per_stack,
        config_.device.stacks, "mapping scratch total bytes");
    stats_.controller_dram_budget_bytes = config_.host.ctrl_dram_bytes;
    stats_.controller_dram_budget_bytes_per_stack =
        config_.host.ctrl_dram_bytes / config_.device.stacks;
    stats_.write_buffer_capacity_bytes = write_buffer_dram.total_bytes;
    stats_.write_buffer_capacity_bytes_per_stack =
        write_buffer_dram.bytes_per_stack;
    stats_.mapping_table_bytes = required_ctrl_dram_bytes;
    stats_.mapping_table_bytes_per_stack = mapping_table_bytes_per_stack_;
    stats_.mapping_table_pages_per_stack = mapping_table_pages_per_stack_;
    if (config_.host.mapping_mode == MappingMode::RawPhysical) {
        // The exposed-address-space mode carries no L2P state of any kind:
        // no resident table, no directory, no cache. Controller DRAM holds
        // only the configured write buffer.
        if (config_.host.write_coalescing_enabled) {
            throw std::runtime_error(
                "HBF raw-physical mode requires write coalescing disabled: "
                "buffered flushes would reintroduce FTL allocation");
        }
        mapping_table_pages_per_stack_ = 0;
        mapping_table_bytes_per_stack_ = 0;
        stats_.mapping_table_bytes = 0;
        stats_.mapping_table_bytes_per_stack = 0;
        stats_.mapping_table_pages_per_stack = 0;
        if (config_.host.ctrl_dram_bytes == 0) {
            config_.host.ctrl_dram_bytes = write_buffer_dram.total_bytes;
            stats_.controller_dram_budget_bytes = config_.host.ctrl_dram_bytes;
            stats_.controller_dram_budget_bytes_per_stack =
                config_.host.ctrl_dram_bytes / config_.device.stacks;
        } else if (
            config_.host.ctrl_dram_bytes / config_.device.stacks <
            write_buffer_dram.bytes_per_stack) {
            throw std::runtime_error(
                "HBF raw-physical mode cannot hold the configured write buffer "
                "in every stack partition");
        }
    } else if (config_.host.mapping_mode == MappingMode::FullResident) {
        const auto required_per_stack = checked_add(
            mapping_table_bytes_per_stack_ + scratch_per_stack + copy_per_stack,
            write_buffer_dram.bytes_per_stack,
            "HBF resident mapping plus write-buffer DRAM per stack");
        if (config_.host.ctrl_dram_bytes != 0 &&
            config_.host.ctrl_dram_bytes / config_.device.stacks < required_per_stack) {
            throw std::runtime_error(
                "HBF ctrl_dram_bytes cannot hold the complete resident L2P "
                "table and configured write buffer in every stack partition");
        }
        if (config_.host.ctrl_dram_bytes == 0) {
            config_.host.ctrl_dram_bytes = checked_add(
                required_ctrl_dram_bytes + stats_.mapping_scratch_capacity_bytes + stats_.host_gc_buffer_bytes,
                write_buffer_dram.total_bytes,
                "HBF derived resident controller-DRAM budget");
            stats_.controller_dram_budget_bytes = config_.host.ctrl_dram_bytes;
            stats_.controller_dram_budget_bytes_per_stack =
                config_.host.ctrl_dram_bytes / config_.device.stacks;
        }
        stats_.resident_mapping_table_bytes = required_ctrl_dram_bytes;
        stats_.resident_mapping_table_bytes_per_stack =
            mapping_table_bytes_per_stack_;
        stats_.resident_mapping_pages_per_stack =
            mapping_table_pages_per_stack_;
    } else {
        if (config_.host.ctrl_dram_bytes == 0) {
            throw std::runtime_error(
                "HBF cached mapping requires an explicit positive "
                "ctrl_dram_bytes budget");
        }
        stats_.mapping_directory_entry_bytes = checked_add(
            config_.host.mapping_directory_entry_bytes,
            config_.host.mapping_cache_layout == MappingCacheLayout::Entry ? 8 : 0,
            "mapping directory pointer and persisted epoch");
        if (config_.host.mapping_cache_layout == MappingCacheLayout::Entry)
            mapping_checkpointed_through_ns_.assign(
                mapping_table_pages_per_stack_ * config_.device.stacks, -1.0);
        stats_.mapping_directory_bytes_per_stack = checked_mul(
            mapping_table_pages_per_stack_,
            stats_.mapping_directory_entry_bytes,
            "HBF mapping-directory bytes per stack");
        stats_.mapping_directory_bytes = checked_mul(
            stats_.mapping_directory_bytes_per_stack,
            config_.device.stacks,
            "HBF mapping-directory total bytes");
        const auto ctrl_dram_bytes_per_stack =
            config_.host.ctrl_dram_bytes / config_.device.stacks;
        auto minimum_cached_bytes_per_stack = checked_add(
            stats_.mapping_directory_bytes_per_stack,
            write_buffer_dram.bytes_per_stack,
            "HBF cached mapping directory plus write-buffer DRAM per stack");
        minimum_cached_bytes_per_stack = checked_add(
            minimum_cached_bytes_per_stack,
            config_.device.page_size_bytes + scratch_per_stack + copy_per_stack,
            "HBF minimum cached-mapping DRAM per stack");
        if (ctrl_dram_bytes_per_stack < minimum_cached_bytes_per_stack) {
            throw std::runtime_error(
                "HBF cached mapping requires its complete mapping-page "
                "directory, configured write buffer, and at least one cache "
                "page per stack");
        }
        const auto cache_payload_budget =
            (ctrl_dram_bytes_per_stack -
             stats_.mapping_directory_bytes_per_stack -
             write_buffer_dram.bytes_per_stack - scratch_per_stack - copy_per_stack);
        const auto full_record_bytes = checked_add(
            config_.device.page_size_bytes, config_.host.mapping_cache_tag_bytes,
            "mapping cache full record bytes");
        mapping_cache_pages_per_stack_ = cache_payload_budget / full_record_bytes;
        if (mapping_cache_pages_per_stack_ == 0) {
            throw std::runtime_error("mapping cache cannot hold a full record including tags");
        }
        const auto minimum_record_bytes = config_.host.mapping_cache_layout == MappingCacheLayout::Page ?
            full_record_bytes : config_.host.mapping_cache_tag_bytes +
                (config_.host.mapping_cache_layout == MappingCacheLayout::Entry ? 8 : 24);
        mapping_cache_record_limit_per_stack_ = cache_payload_budget / minimum_record_bytes;
        stats_.mapping_cache_pages_per_stack =
            mapping_cache_pages_per_stack_;
        stats_.mapping_cache_capacity_bytes_per_stack = checked_mul(
            mapping_cache_pages_per_stack_,
            full_record_bytes,
            "HBF mapping-cache bytes per stack");
        stats_.mapping_cache_capacity_bytes = checked_mul(
            stats_.mapping_cache_capacity_bytes_per_stack,
            config_.device.stacks,
            "HBF mapping-cache total bytes");
    }
    stats_.mapping_dram_resources = config_.device.stacks;

    channels_.resize(static_cast<std::size_t>(total_channels));
    dies_.resize(static_cast<std::size_t>(total_dies));
    planes_.resize(static_cast<std::size_t>(total_planes));
    gc_headroom_by_plane_.resize(planes_.size());
    gc_headroom_by_stack_.resize(config_.device.stacks);
    for (auto& cache : gc_headroom_by_stack_)
        cache.dirty_planes.reserve(planes_per_stack);
    logic_dies_.resize(config_.device.stacks);
    mapping_memory_issue_by_stack_.resize(config_.device.stacks);
    local_mapping_bus_by_stack_.resize(config_.device.stacks);
    mapping_sram_lru_by_stack_.resize(config_.device.stacks);
    mapping_sram_by_stack_.resize(config_.device.stacks);
    mapping_sram_issue_by_stack_.resize(config_.device.stacks);
    stats_.mapping_sram_reserved_bytes = config_.host.mapping_sram_bytes;
    const auto mapping_workers = config_.host.mapping_compute_workers ?
        config_.host.mapping_compute_workers : config_.device.stacks;
    if ((mapping_is_local() || !config_.host.mapping_host_compute_shared) &&
        mapping_workers % config_.device.stacks != 0)
        throw std::runtime_error("device-local mapping workers must partition equally across stacks");
    mapping_compute_workers_.resize(mapping_workers);
    stats_.mapping_compute_workers = mapping_workers;
    if (mapping_is_local()) stats_.local_mapping_reserved_bytes = config_.host.ctrl_dram_bytes;
    blocks_.resize(static_cast<std::size_t>(total_blocks));
    const auto channel_blocks = total_blocks / total_channels;
    if (channel_blocks % config_.host.zone_size_blocks != 0) {
        throw std::runtime_error("HBF equal zones must divide each channel in whole NAND blocks");
    }
    media_->configure_host_zones(config_.host.zone_size_blocks);
    gc_candidates_by_stack_.resize(config_.device.stacks);
    gc_dirty_blocks_by_stack_.resize(config_.device.stacks);
    managed_wear_by_stack_.resize(config_.device.stacks);
    for (std::size_t stack = 0; stack < config_.device.stacks; ++stack) {
        if (config_.host.auto_gc_enabled && config_.host.mapping_mode != MappingMode::RawPhysical)
            gc_candidates_by_stack_[stack].resize(2 * config_.device.pages_per_block);
        managed_wear_by_stack_[stack][0] = total_blocks / config_.device.stacks;
    }
    for (auto& plane : planes_) {
        plane.subarrays.resize(subarrays_per_plane_);
        plane.media_lanes.resize(config_.device.media_lanes_per_plane);
        plane.page_buffer_banks.resize(config_.device.page_buffer_banks_per_plane);
    }
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        auto& block = blocks_[block_index];
        block.free_pages = config_.device.pages_per_block;
        block.next_page = 0;
        planes_.at(block_plane_index(block_index)).free_blocks.push_back(block_index);
    }
    rebuild_active_plane_counts();
    stats_.free_pages = free_pages_;
    stats_.gc_reserve_pages = checked_mul(
        checked_mul(
            static_cast<std::uint64_t>(config_.host.gc_reserved_free_blocks_per_plane),
            total_planes,
            "HBF GC reserve blocks"),
        config_.device.pages_per_block,
        "HBF GC reserve pages");
    // Provisional logical capacity; static/raw reservations re-derive it and
    // the first logical use freezes it.
    if (config.host.mapping_organization != MappingOrganization::Page)
        mapping_policy_ = std::make_unique<HbfMappingPolicy>(config, config_);
    logical_capacity_pages_ = logical_capacity_pages();
    stats_.logical_capacity_pages = logical_capacity_pages_;
    stats_.logical_capacity_bytes = checked_mul(
        logical_capacity_pages_, config_.device.page_size_bytes, "HBF logical capacity bytes");
    refresh_parallel_stats();
}

bool HbfController::managed_block_role(BlockRole role) {
    return role != BlockRole::StaticReadOnly && role != BlockRole::RawPhysical;
}

std::size_t HbfController::active_planes_in_stack(std::size_t stack) const {
    return active_planes_by_stack_.at(stack);
}

void HbfController::set_block_role(std::size_t block_index, BlockRole role) {
    auto& block = blocks_.at(block_index);
    mark_gc_headroom_dirty(block_plane_index(block_index));
    const bool was_managed = managed_block_role(block.role);
    const bool managed = managed_block_role(role);
    block.role = role;
    if (was_managed == managed) {
        return;
    }
    const auto plane_index = block_plane_index(block_index);
    auto& managed_blocks = managed_blocks_by_plane_.at(plane_index);
    auto& active_planes = active_planes_by_stack_.at(plane_index / planes_per_stack());
    if (managed) {
        if (managed_blocks++ == 0) {
            active_planes++;
        }
    } else if (--managed_blocks == 0) {
        active_planes--;
    }
}

void HbfController::rebuild_active_plane_counts() {
    // Construction and persistent restore rebuild ownership outside the
    // normal allocation hooks. Recompute derived capacity on its next use.
    reset_gc_headroom_cache();
    managed_blocks_by_plane_.assign(planes_.size(), 0);
    active_planes_by_stack_.assign(config_.device.stacks, 0);
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        if (managed_block_role(blocks_[block_index].role)) {
            managed_blocks_by_plane_[block_plane_index(block_index)]++;
        }
    }
    for (std::size_t plane_index = 0; plane_index < planes_.size(); ++plane_index) {
        if (managed_blocks_by_plane_[plane_index] != 0) {
            active_planes_by_stack_[plane_index / planes_per_stack()]++;
        }
    }
}

std::size_t HbfController::scan_active_planes_in_stack(std::size_t stack) const {
    const auto pps = planes_per_stack();
    std::size_t active = 0;
    for (std::size_t plane_index = stack * pps;
         plane_index < (stack + 1) * pps;
         ++plane_index) {
        for (std::uint32_t block = 0; block < config_.device.blocks_per_plane; ++block) {
            const auto role = blocks_.at(
                plane_index * config_.device.blocks_per_plane + block).role;
            if (role != BlockRole::StaticReadOnly &&
                role != BlockRole::RawPhysical) {
                active++;
                break;
            }
        }
    }
    return active;
}

std::uint64_t HbfController::derive_logical_capacity_pages() const {
    // Largest logical page count L (over the whole device, page-striped
    // across stacks so every stack serves at most ceil(L / stacks) pages)
    // such that, per stack,
    //   L_s + M(L_s) <= (usable_blocks - open_blocks) * pages_per_block - 1
    // with M(L_s) = ceil(L_s / mapping_entries_per_page) persistent mapping
    // pages, usable_blocks the managed blocks (static/raw extents excluded)
    // beyond the stack's GC-only reserve, and open_blocks the blocks that
    // can still hold free pages when the foreground stalls.
    //
    // Why this guarantees a victim. The foreground (Data or Mapping role)
    // stalls only when its own write frontier is exhausted and the GC pool
    // minus one block is below the floor (which is at most the reserve), so
    // at most reserve/ppb whole free blocks remain. Once in-flight commits
    // have landed, the only non-free blocks that are not closed are the GC
    // write frontier (one per stack: relocation opens a new block only
    // when no GC block has free pages), the Data and Mapping frontiers (one
    // each per plane with managed blocks; the stalled role's are exhausted,
    // but a fresh compact image leaves both partially filled), and the
    // pinned wear-leveling cold block. Every other usable block is closed,
    // so
    //   closed_blocks >= usable_blocks - open_blocks.
    // Every valid page belongs to exactly one live LPN or persisted mapping
    // VPN, so valid_pages <= L_s + M(L_s) < closed_blocks * pages_per_block:
    // some closed block holds an invalid page. That victim keeps fewer than
    // pages_per_block live pages and the reserve, which the foreground
    // never breaches, holds them (see gc_reserve_requirement_pages).
    const auto pps = planes_per_stack();
    const auto ppb = static_cast<std::uint64_t>(config_.device.pages_per_block);
    const auto entries = static_cast<std::uint64_t>(config_.host.mapping_entries_per_page);
    const bool ftl = config_.host.mapping_mode != MappingMode::RawPhysical;
    const auto reserve = ftl ? static_cast<std::uint64_t>(
        config_.host.gc_reserved_free_blocks_per_plane) : 0;
    std::uint64_t minimum_stack_pages = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t stack = 0; stack < config_.device.stacks; ++stack) {
        std::uint64_t usable_blocks = 0;
        std::uint64_t active_planes = 0;
        for (std::size_t plane_index = stack * pps;
             plane_index < (stack + 1) * pps;
             ++plane_index) {
            std::uint64_t managed = 0;
            for (std::uint32_t block = 0; block < config_.device.blocks_per_plane; ++block) {
                const auto role = blocks_.at(
                    plane_index * config_.device.blocks_per_plane + block).role;
                if (role != BlockRole::StaticReadOnly &&
                    role != BlockRole::RawPhysical) {
                    managed++;
                }
            }
            if (managed != 0) {
                usable_blocks += managed;
                active_planes++;
            }
        }
        const auto reserve_blocks = checked_mul(
            reserve, active_planes, "HBF logical capacity reserve blocks");
        usable_blocks = usable_blocks > reserve_blocks ?
            usable_blocks - reserve_blocks : 0;
        // Open blocks: the stack's GC frontier, the Data and Mapping
        // frontiers of every plane with managed blocks, and the cold block.
        const std::uint64_t open_blocks = !ftl ? 0 : checked_add(
            checked_add(
                1,
                checked_mul(2, active_planes, "HBF logical capacity open blocks"),
                "HBF logical capacity open blocks"),
            config_.host.static_wear_leveling_erase_gap != 0 ? 1 : 0,
            "HBF logical capacity cold frontier");
        if (usable_blocks <= open_blocks) {
            return 0;
        }
        auto budget = checked_mul(
            usable_blocks - open_blocks, ppb, "HBF logical capacity budget");
        if (!ftl) {
            minimum_stack_pages = std::min(minimum_stack_pages, budget);
            continue;
        }
        budget -= 1;
        // Largest L with L + ceil(L / entries) <= budget.
        std::uint64_t pages = budget / (entries + 1) * entries;
        const auto fits = [&](std::uint64_t candidate) {
            return candidate + checked_ceil_div(
                candidate, entries, "HBF logical capacity mapping pages") <= budget;
        };
        while (pages != 0 && !fits(pages)) {
            --pages;
        }
        while (fits(pages + 1)) {
            ++pages;
        }
        minimum_stack_pages = std::min(minimum_stack_pages, pages);
    }
    return checked_mul(
        minimum_stack_pages,
        static_cast<std::uint64_t>(config_.device.stacks),
        "HBF logical capacity pages");
}

std::uint64_t HbfController::logical_capacity_pages() const {
    if (mapping_policy_) return mapping_policy_->capacity_pages();
    if (logical_capacity_frozen_) {
        return logical_capacity_pages_;
    }
    const auto derived = derive_logical_capacity_pages();
    if (config_.host.logical_capacity_bytes == 0) {
        return derived;
    }
    if (config_.host.logical_capacity_bytes % config_.device.page_size_bytes != 0) {
        throw std::runtime_error(
            "HBF logical_capacity_bytes must be a multiple of page_size_bytes");
    }
    const auto pages = config_.host.logical_capacity_bytes / config_.device.page_size_bytes;
    if (pages > derived) {
        throw std::runtime_error(
            "HBF logical_capacity_bytes exceeds the maximum safe logical capacity of " +
            std::to_string(derived) + " pages for this geometry and reservations");
    }
    return pages;
}

void HbfController::resolve_logical_capacity() {
    if (logical_capacity_frozen_) {
        return;
    }
    // Static/raw extents may have made planes inert; the stack's remaining
    // reserve must still hold one worst-case victim.
    const auto required = gc_reserve_requirement_pages();
    for (std::size_t stack = 0; required != 0 && stack < config_.device.stacks; ++stack) {
        const auto reserve_pages = gc_reserve_pages(stack);
        if (reserve_pages < required) {
            throw std::runtime_error(
                "HBF stack " + std::to_string(stack) + " keeps a GC-only reserve "
                "of " + std::to_string(reserve_pages) + " pages after its "
                "static/raw reservations, below the " + std::to_string(required) +
                " pages one worst-case victim needs (raise "
                "gc_reserved_free_blocks_per_plane or leave more planes with "
                "managed blocks)");
        }
    }
    const auto capacity = logical_capacity_pages();
    if (capacity == 0) {
        throw std::runtime_error(
            "HBF geometry cannot host any logical page: every stack needs "
            "managed blocks beyond gc_reserved_free_blocks_per_plane per "
            "plane for one GC frontier, one Data and one Mapping frontier "
            "per plane" +
            std::string(config_.host.static_wear_leveling_erase_gap != 0 ?
                ", one wear-leveling cold block" : "") +
            ", and at least one closed block");
    }
    logical_capacity_pages_ = capacity;
    logical_capacity_frozen_ = true;
    stats_.logical_capacity_pages = logical_capacity_pages_;
    stats_.logical_capacity_bytes = checked_mul(
        logical_capacity_pages_, config_.device.page_size_bytes, "HBF logical capacity bytes");
}

void HbfController::require_logical_capacity(std::uint64_t last_lpn) {
    resolve_logical_capacity();
    if (last_lpn >= logical_capacity_pages_) {
        throw std::runtime_error(
            "HBF logical page " + std::to_string(last_lpn) +
            " is beyond the logical capacity of " +
            std::to_string(logical_capacity_pages_) + " pages (" +
            std::to_string(stats_.logical_capacity_bytes) +
            " bytes; hbf-logical-capacity-bytes)");
    }
}

HbfAddress HbfController::decode(std::uint64_t addr) const {
    if (addr / config_.device.page_size_bytes >= total_pages_) {
        throw std::runtime_error("HBF physical byte address is out of range");
    }
    HbfAddress decoded;
    decoded.offset = addr % config_.device.page_size_bytes;
    std::uint64_t unit = addr / config_.device.page_size_bytes;
    decoded.page = static_cast<std::uint32_t>(unit % config_.device.pages_per_block);
    unit /= config_.device.pages_per_block;
    decoded.block = static_cast<std::uint32_t>(unit % config_.device.blocks_per_plane);
    unit /= config_.device.blocks_per_plane;
    decoded.plane = static_cast<std::uint32_t>(unit % config_.device.planes_per_die);
    unit /= config_.device.planes_per_die;
    decoded.die = static_cast<std::uint32_t>(unit % config_.device.dies_per_channel);
    unit /= config_.device.dies_per_channel;
    decoded.channel = static_cast<std::uint32_t>(unit % config_.device.channels_per_stack);
    unit /= config_.device.channels_per_stack;
    decoded.stack = static_cast<std::uint32_t>(unit % config_.device.stacks);
    return decoded;
}

std::uint64_t HbfController::encode(const HbfAddress& addr) const {
    return encode_ppn(addr) * config_.device.page_size_bytes + addr.offset;
}

std::uint64_t HbfController::schedule_commit(
    std::size_t stack,
    double at_ns,
    std::function<void()> action) {
    if (!std::isfinite(at_ns) || at_ns < 0.0) {
        throw std::runtime_error("HBF state commit time must be finite and non-negative");
    }
    if (stack >= config_.device.stacks) {
        throw std::runtime_error("HBF state commit stack is out of range");
    }
    const auto sequence = next_commit_sequence_++;
    insert_pending_commit(
        at_ns,
        sequence,
        PendingCommit{.stack = stack, .action = std::move(action)});
    return sequence;
}

void HbfController::insert_pending_commit(
    double at_ns,
    std::uint64_t sequence,
    PendingCommit commit) {
    const auto key = std::make_pair(at_ns, sequence);
    const auto stack = commit.stack;
    if (stack >= pending_commit_keys_by_stack_.size()) {
        throw std::runtime_error("HBF pending commit names an unknown stack");
    }
    if (!pending_commits_.emplace(key, std::move(commit)).second ||
        !pending_commit_keys_by_stack_[stack].insert(key).second ||
        !pending_commit_time_by_sequence_.emplace(sequence, at_ns).second) {
        throw std::runtime_error("HBF pending commit sequence collided");
    }
}

std::function<void()> HbfController::take_pending_commit(
    std::map<std::pair<double, std::uint64_t>, PendingCommit>::iterator
        commit) {
    auto action = std::move(commit->second.action);
    const auto key = commit->first;
    const auto stack = commit->second.stack;
    if (pending_commit_keys_by_stack_.at(stack).erase(key) != 1 ||
        pending_commit_time_by_sequence_.erase(key.second) != 1) {
        throw std::runtime_error("HBF pending commit index lost an entry");
    }
    pending_commits_.erase(commit);
    return action;
}

void HbfController::apply_selected_commits_through(
    const std::unordered_set<std::uint64_t>& sequences,
    double at_ns) {
    if (sequences.empty()) {
        return;
    }
    // Resolve the selected sequences to their keys first: applying one
    // commit may enqueue or apply others, and the keys must run in time
    // order regardless of the caller's set iteration order.
    std::vector<std::pair<double, std::uint64_t>> keys;
    keys.reserve(sequences.size());
    for (const auto sequence : sequences) {
        const auto time = pending_commit_time_by_sequence_.find(sequence);
        if (time != pending_commit_time_by_sequence_.end() &&
            time->second <= at_ns) {
            keys.emplace_back(time->second, sequence);
        }
    }
    std::sort(keys.begin(), keys.end());
    for (const auto& key : keys) {
        const auto commit = pending_commits_.find(key);
        if (commit == pending_commits_.end()) {
            continue;
        }
        auto action = take_pending_commit(commit);
        action();
    }
}

void HbfController::apply_commits_through(double at_ns) {
    while (!pending_commits_.empty() && pending_commits_.begin()->first.first <= at_ns) {
        auto action = take_pending_commit(pending_commits_.begin());
        action();
    }
}

void HbfController::apply_stack_commits_through(std::size_t stack, double at_ns) {
    auto& keys = pending_commit_keys_by_stack_.at(stack);
    while (!keys.empty() && keys.begin()->first <= at_ns) {
        const auto commit = pending_commits_.find(*keys.begin());
        if (commit == pending_commits_.end()) {
            throw std::runtime_error(
                "HBF pending commit index names a missing commit");
        }
        auto action = take_pending_commit(commit);
        action();
    }
}

void HbfController::apply_all_commits() {
    while (!pending_commits_.empty()) {
        const double at_ns = pending_commits_.begin()->first.first;
        apply_commits_through(at_ns);
    }
}

bool HbfController::advance_to_next_commit(double& at_ns, std::size_t stack) {
    const auto& keys = pending_commit_keys_by_stack_.at(stack);
    if (keys.empty()) {
        return false;
    }
    at_ns = std::max(at_ns, keys.begin()->first);
    apply_stack_commits_through(stack, at_ns);
    return true;
}

std::optional<std::uint64_t> HbfController::visible_lpn_at(
    std::uint64_t lpn,
    double at_ns,
    const CompactReadAddress* address) const {
    std::optional<std::uint64_t> visible;
    if (const auto base = lpn_to_ppn_.find(lpn); base != lpn_to_ppn_.end()) {
        const auto page = programmed_pages_.find(base->second);
        const auto block_index = static_cast<std::size_t>(
            base->second / config_.device.pages_per_block);
        const auto& block = blocks_.at(block_index);
        if (page != programmed_pages_.end() &&
            page->second.status == PageStatus::Valid &&
            page->second.owner == PageOwner::Logical &&
            page->second.block_epoch == block.epoch &&
            !block.erase_pending) {
            visible = base->second;
        }
    }
    if (!visible) {
        visible = compact_lpn_ppn(lpn, address);
    }
    const auto pending = pending_lpn_updates_.find(lpn);
    if (pending == pending_lpn_updates_.end()) {
        return visible;
    }
    const PendingMappingUpdate* newest = nullptr;
    for (const auto& update : pending->second) {
        const auto block_index = static_cast<std::size_t>(
            update.new_ppn / config_.device.pages_per_block);
        if (blocks_.at(block_index).epoch == update.block_epoch &&
            !blocks_.at(block_index).erase_pending &&
            update.commit_ns <= at_ns &&
            (newest == nullptr || std::tie(update.commit_ns, update.sequence) >
                std::tie(newest->commit_ns, newest->sequence))) {
            newest = &update;
        }
    }
    return newest == nullptr ? visible : std::optional<std::uint64_t>{newest->new_ppn};
}

void HbfController::wait_for_prior_lpn_commit(
    std::uint64_t lpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    bool wait_for_relocations) {
    double dependency_ready_ns = 0.0;
    if (const auto observed = materialized_ready_by_lpn_.find(lpn);
        observed != materialized_ready_by_lpn_.end()) {
        dependency_ready_ns = observed->second;
    }
    std::unordered_set<std::uint64_t> commit_sequences;
    const auto pending = pending_lpn_updates_.find(lpn);
    if (pending != pending_lpn_updates_.end()) {
        for (const auto& update : pending->second) {
            if (update.relocation && !wait_for_relocations) {
                // The old copy stays intact and carries the same bytes until
                // the victim's erase is issued; whichever publication commits
                // later already discards or invalidates the loser.
                continue;
            }
            // Per-LPN ordering survives destruction of the target: a later
            // read waits for the prior write callback and its target erase.
            dependency_ready_ns = std::max({
                dependency_ready_ns,
                update.commit_ns,
                update.destructive_ready_ns,
            });
            commit_sequences.insert(update.program_commit_sequence);
            commit_sequences.insert(update.sequence);
            const auto block_index = static_cast<std::size_t>(
                update.new_ppn / config_.device.pages_per_block);
            add_pending_block_transition_commits(
                block_index, commit_sequences);
        }
    }
    const double required_ns = std::max(at_ns, dependency_ready_ns);
    if (required_ns > at_ns) {
        if (spans != nullptr) {
            trace_wait(
                spans,
                logic_entity(static_cast<std::uint32_t>(stack_for_lpn(lpn))),
                at_ns,
                required_ns,
                "wait_prior_lpn_commit");
        }
        breakdown.scheduler_queue_wait_ns += required_ns - at_ns;
        at_ns = required_ns;
    }
    const auto stack = stack_for_lpn(lpn);
    apply_selected_commits_through(commit_sequences, at_ns);
    if (dependency_ready_ns != 0.0) {
        materialized_ready_by_lpn_[lpn] = std::max(
            materialized_ready_by_lpn_[lpn], dependency_ready_ns);
    }
    state_observation_by_stack_.at(stack) = std::max(
        state_observation_by_stack_.at(stack), at_ns);
}

void HbfController::wait_for_pending_vpn_erase(
    std::uint64_t mapping_vpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    double dependency_ready_ns = 0.0;
    if (const auto observed = materialized_ready_by_vpn_.find(mapping_vpn);
        observed != materialized_ready_by_vpn_.end()) {
        dependency_ready_ns = observed->second;
    }
    std::unordered_set<std::uint64_t> commit_sequences;
    bool has_destructive_dependency = false;
    const auto pending = pending_vpn_updates_.find(mapping_vpn);
    if (pending != pending_vpn_updates_.end()) {
        for (const auto& update : pending->second) {
            has_destructive_dependency |= update.destructive_ready_ns != 0.0;
            dependency_ready_ns = std::max(
                dependency_ready_ns, update.destructive_ready_ns);
            commit_sequences.insert(update.program_commit_sequence);
            commit_sequences.insert(update.sequence);
            const auto block_index = static_cast<std::size_t>(
                update.new_ppn / config_.device.pages_per_block);
            add_pending_block_transition_commits(
                block_index, commit_sequences);
        }
    }
    if (!has_destructive_dependency && dependency_ready_ns == 0.0) {
        return;
    }
    const double required_ns = std::max(at_ns, dependency_ready_ns);
    const auto stack = stack_for_vpn(mapping_vpn);
    if (required_ns > at_ns) {
        if (spans != nullptr) {
            trace_wait(
                spans,
                logic_entity(static_cast<std::uint32_t>(stack)),
                at_ns,
                required_ns,
                "wait_pending_mapping_block_erase");
        }
        breakdown.scheduler_queue_wait_ns += required_ns - at_ns;
        at_ns = required_ns;
    }
    apply_selected_commits_through(commit_sequences, at_ns);
    materialized_ready_by_vpn_[mapping_vpn] = std::max(
        materialized_ready_by_vpn_[mapping_vpn], dependency_ready_ns);
    state_observation_by_stack_.at(stack) = std::max(
        state_observation_by_stack_.at(stack), at_ns);
}

bool HbfController::current_ppn_has_pending_transition(
    std::uint64_t lpn, const CompactReadAddress* address) const {
    std::optional<std::uint64_t> current;
    if (const auto mapping = lpn_to_ppn_.find(lpn);
        mapping != lpn_to_ppn_.end()) {
        current = mapping->second;
    } else {
        current = compact_lpn_ppn(lpn, address);
    }
    if (!current) {
        return false;
    }
    const auto block_index = static_cast<std::size_t>(
        *current / config_.device.pages_per_block);
    return blocks_.at(block_index).erase_pending ||
        pending_block_transitions_.contains(block_index);
}

void HbfController::wait_for_pending_block_erase(
    std::uint64_t ppn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    // Only a raw (host-issued) erase destroys data the mapping still names;
    // a GC reclaim is issued after every live page has been published
    // elsewhere and is never waited for here.
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.device.pages_per_block);
    if (!blocks_.at(block_index).erase_pending) {
        return;
    }
    double dependency_ready_ns = materialized_ready_by_block_.at(block_index);
    std::unordered_set<std::uint64_t> commit_sequences;
    const auto pending = pending_block_transitions_.find(block_index);
    if (pending != pending_block_transitions_.end()) {
        dependency_ready_ns = std::max(
            dependency_ready_ns, pending->second.finish_ns);
        add_pending_block_transition_commits(
            block_index, commit_sequences);
    }
    if (dependency_ready_ns == 0.0) {
        return;
    }
    const auto stack = stack_of_block(block_index);
    if (dependency_ready_ns > at_ns) {
        if (spans != nullptr) {
            trace_wait(
                spans,
                logic_entity(static_cast<std::uint32_t>(stack)),
                at_ns,
                dependency_ready_ns,
                "wait_target_block_erase");
        }
        breakdown.scheduler_queue_wait_ns += dependency_ready_ns - at_ns;
        at_ns = dependency_ready_ns;
    }
    apply_selected_commits_through(commit_sequences, at_ns);
    materialized_ready_by_block_.at(block_index) = std::max(
        materialized_ready_by_block_.at(block_index), dependency_ready_ns);
    state_observation_by_stack_.at(stack) = std::max(
        state_observation_by_stack_.at(stack), at_ns);
}

void HbfController::add_pending_block_transition_commits(
    std::size_t block_index,
    std::unordered_set<std::uint64_t>& sequences) const {
    const auto pending = pending_block_transitions_.find(block_index);
    if (pending == pending_block_transitions_.end()) {
        return;
    }
    sequences.insert(
        pending->second.dependency_sequences.begin(),
        pending->second.dependency_sequences.end());
    sequences.insert(pending->second.commit_sequence);
}

void HbfController::tag_pending_mapping_updates_for_erase(
    std::size_t block_index,
    std::uint64_t retired_block_epoch,
    double ready_ns) {
    const auto tag = [this, block_index, retired_block_epoch, ready_ns](auto& table) {
        for (auto& [_, updates] : table) {
            for (auto& update : updates) {
                if (update.new_ppn / config_.device.pages_per_block == block_index &&
                    update.block_epoch == retired_block_epoch) {
                    update.destructive_ready_ns = std::max(
                        update.destructive_ready_ns, ready_ns);
                }
            }
        }
    };
    tag(pending_lpn_updates_);
    tag(pending_vpn_updates_);
}

void HbfController::retire_mapping_update_tombstones(
    std::size_t block_index,
    double ready_ns) {
    const auto retire = [this, block_index, ready_ns](
                            auto& table, auto& materialized_ready) {
        for (auto entry = table.begin(); entry != table.end();) {
            const bool retires_entry = std::any_of(
                entry->second.begin(),
                entry->second.end(),
                [this, block_index, ready_ns](const PendingMappingUpdate& update) {
                    return update.new_ppn / config_.device.pages_per_block == block_index &&
                        update.destructive_ready_ns != 0.0 &&
                        update.destructive_ready_ns <= ready_ns &&
                        update.commit_ns <= ready_ns;
                });
            std::erase_if(entry->second, [this, block_index, ready_ns](
                              const PendingMappingUpdate& update) {
                return update.new_ppn / config_.device.pages_per_block == block_index &&
                    update.destructive_ready_ns != 0.0 &&
                    update.destructive_ready_ns <= ready_ns &&
                    update.commit_ns <= ready_ns;
            });
            if (retires_entry) {
                materialized_ready[entry->first] = std::max(
                    materialized_ready[entry->first], ready_ns);
            }
            if (entry->second.empty()) {
                entry = table.erase(entry);
            } else {
                ++entry;
            }
        }
    };
    retire(pending_lpn_updates_, materialized_ready_by_lpn_);
    retire(pending_vpn_updates_, materialized_ready_by_vpn_);
}

std::optional<std::uint64_t> HbfController::visible_mapping_vpn_at(
    std::uint64_t mapping_vpn,
    double at_ns) const {
    std::optional<std::uint64_t> visible;
    if (const auto base = mapping_vpn_to_ppn_.find(mapping_vpn);
        base != mapping_vpn_to_ppn_.end()) {
        const auto page = programmed_pages_.find(base->second);
        const auto block_index = static_cast<std::size_t>(
            base->second / config_.device.pages_per_block);
        const auto& block = blocks_.at(block_index);
        if (page != programmed_pages_.end() &&
            page->second.status == PageStatus::Valid &&
            page->second.owner == PageOwner::Mapping &&
            page->second.block_epoch == block.epoch &&
            !block.erase_pending) {
            visible = base->second;
        }
    }
    if (!visible) {
        visible = compact_mapping_ppn(mapping_vpn);
    }
    const auto pending = pending_vpn_updates_.find(mapping_vpn);
    if (pending == pending_vpn_updates_.end()) {
        return visible;
    }
    const PendingMappingUpdate* newest = nullptr;
    for (const auto& update : pending->second) {
        const auto block_index = static_cast<std::size_t>(
            update.new_ppn / config_.device.pages_per_block);
        if (blocks_.at(block_index).epoch == update.block_epoch &&
            !blocks_.at(block_index).erase_pending &&
            update.commit_ns <= at_ns &&
            (newest == nullptr || std::tie(update.commit_ns, update.sequence) >
                std::tie(newest->commit_ns, newest->sequence))) {
            newest = &update;
        }
    }
    return newest == nullptr ? visible : std::optional<std::uint64_t>{newest->new_ppn};
}

void HbfController::schedule_lpn_mapping_commit(
    std::uint64_t lpn,
    std::uint64_t new_ppn,
    double commit_ns,
    std::uint64_t program_commit_sequence,
    std::optional<std::uint64_t> expected_old_ppn) {
    const auto sequence = next_commit_sequence_++;
    const auto block_index = static_cast<std::size_t>(
        new_ppn / config_.device.pages_per_block);
    const auto block_epoch = blocks_.at(block_index).epoch;
    blocks_.at(block_index).pending_mapping_publications++;
    pending_lpn_updates_[lpn].push_back(PendingMappingUpdate{
        .sequence = sequence,
        .program_commit_sequence = program_commit_sequence,
        .commit_ns = commit_ns,
        .new_ppn = new_ppn,
        .block_epoch = block_epoch,
        .relocation = expected_old_ppn.has_value(),
    });
    insert_pending_commit(
        commit_ns,
        sequence,
        PendingCommit{
        .stack = stack_for_lpn(lpn),
        .action = [this, lpn, new_ppn, block_index, block_epoch, sequence,
                      commit_ns, expected_old_ppn]() {
            const auto page = programmed_pages_.find(new_ppn);
            const bool target_survived =
                blocks_.at(block_index).epoch == block_epoch &&
                page != programmed_pages_.end() &&
                page->second.block_epoch == block_epoch &&
                page->second.status == PageStatus::Valid &&
                page->second.owner == PageOwner::Logical;
            bool source_is_current = true;
            if (expected_old_ppn) {
                if (const auto current = lpn_to_ppn_.find(lpn);
                    current != lpn_to_ppn_.end()) {
                    source_is_current = current->second == *expected_old_ppn;
                } else {
                    source_is_current =
                        compact_lpn_ppn(lpn) == expected_old_ppn;
                }
            }
            if (target_survived && source_is_current) {
                const auto current = lpn_to_ppn_.find(lpn);
                if (current != lpn_to_ppn_.end() && current->second != new_ppn) {
                    invalidate_ppn(current->second);
                } else if (current == lpn_to_ppn_.end() &&
                           compact_lpn_ppn(lpn)) {
                    retire_compact_page(lpn, PageOwner::Logical);
                }
                lpn_to_ppn_[lpn] = new_ppn;
            } else if (target_survived && expected_old_ppn) {
                // GC copied a snapshot. A foreground write that committed
                // first is authoritative; retire the stale copy instead of
                // publishing old payload over the newer version.
                invalidate_ppn(new_ppn);
            }
            if (blocks_.at(block_index).epoch == block_epoch) {
                auto& pending_publications =
                    blocks_.at(block_index).pending_mapping_publications;
                if (pending_publications == 0) {
                    throw std::runtime_error(
                        "HBF LPN publication lost its block ownership pin");
                }
                pending_publications--;
            }
            auto pending = pending_lpn_updates_.find(lpn);
            if (pending != pending_lpn_updates_.end()) {
                std::erase_if(
                    pending->second,
                    [sequence](const PendingMappingUpdate& update) {
                        return update.sequence == sequence &&
                            update.destructive_ready_ns <= update.commit_ns;
                    });
                if (pending->second.empty()) {
                    pending_lpn_updates_.erase(pending);
                }
            }
            materialized_ready_by_lpn_[lpn] = std::max(
                materialized_ready_by_lpn_[lpn], commit_ns);
        }});
}

void HbfController::schedule_vpn_mapping_commit(
    std::uint64_t mapping_vpn,
    std::uint64_t new_ppn,
    double commit_ns,
    std::uint64_t program_commit_sequence,
    std::optional<std::uint64_t> expected_old_ppn) {
    const auto sequence = next_commit_sequence_++;
    const auto block_index = static_cast<std::size_t>(
        new_ppn / config_.device.pages_per_block);
    const auto block_epoch = blocks_.at(block_index).epoch;
    blocks_.at(block_index).pending_mapping_publications++;
    pending_vpn_updates_[mapping_vpn].push_back(PendingMappingUpdate{
        .sequence = sequence,
        .program_commit_sequence = program_commit_sequence,
        .commit_ns = commit_ns,
        .new_ppn = new_ppn,
        .block_epoch = block_epoch,
        .relocation = expected_old_ppn.has_value(),
    });
    insert_pending_commit(
        commit_ns,
        sequence,
        PendingCommit{
        .stack = stack_for_vpn(mapping_vpn),
        .action = [this, mapping_vpn, new_ppn, block_index, block_epoch,
                      sequence, commit_ns, expected_old_ppn]() {
            const auto page = programmed_pages_.find(new_ppn);
            const bool target_survived =
                blocks_.at(block_index).epoch == block_epoch &&
                page != programmed_pages_.end() &&
                page->second.block_epoch == block_epoch &&
                page->second.status == PageStatus::Valid &&
                page->second.owner == PageOwner::Mapping;
            bool source_is_current = true;
            if (expected_old_ppn) {
                if (const auto current = mapping_vpn_to_ppn_.find(mapping_vpn);
                    current != mapping_vpn_to_ppn_.end()) {
                    source_is_current = current->second == *expected_old_ppn;
                } else {
                    source_is_current =
                        compact_mapping_ppn(mapping_vpn) == expected_old_ppn;
                }
            }
            if (target_survived && source_is_current) {
                const auto current = mapping_vpn_to_ppn_.find(mapping_vpn);
                if (current != mapping_vpn_to_ppn_.end() && current->second != new_ppn) {
                    invalidate_ppn(current->second);
                } else if (
                    current == mapping_vpn_to_ppn_.end() &&
                    compact_mapping_ppn(mapping_vpn)) {
                    retire_compact_page(
                        metadata_lpn(mapping_vpn),
                        PageOwner::Mapping);
                }
                mapping_vpn_to_ppn_[mapping_vpn] = new_ppn;
            } else if (target_survived && expected_old_ppn) {
                invalidate_ppn(new_ppn);
            }
            if (blocks_.at(block_index).epoch == block_epoch) {
                auto& pending_publications =
                    blocks_.at(block_index).pending_mapping_publications;
                if (pending_publications == 0) {
                    throw std::runtime_error(
                        "HBF VPN publication lost its block ownership pin");
                }
                pending_publications--;
            }
            auto pending = pending_vpn_updates_.find(mapping_vpn);
            if (pending != pending_vpn_updates_.end()) {
                std::erase_if(
                    pending->second,
                    [sequence](const PendingMappingUpdate& update) {
                        return update.sequence == sequence &&
                            update.destructive_ready_ns <= update.commit_ns;
                    });
                if (pending->second.empty()) {
                    pending_vpn_updates_.erase(pending);
                }
            }
            materialized_ready_by_vpn_[mapping_vpn] = std::max(
                materialized_ready_by_vpn_[mapping_vpn], commit_ns);
        }});
}

std::uint64_t HbfController::schedule_media_program_commit(
    std::uint64_t ppn,
    std::uint64_t lpn,
    PageOwner owner,
    double commit_ns) {
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.device.pages_per_block);
    const auto block_epoch = blocks_.at(block_index).epoch;
    const auto page = programmed_pages_.find(ppn);
    if (page == programmed_pages_.end() ||
        page->second.status != PageStatus::Erased ||
        page->second.block_epoch != block_epoch) {
        throw std::runtime_error(
            "HBF media program commit was scheduled without an erased reservation");
    }
    return schedule_commit(
        stack_of_block(block_index),
        commit_ns,
        [this, ppn, lpn, owner, block_index, block_epoch, commit_ns]() {
            // An erase with a newer epoch is authoritative. A stale media
            // completion becomes a no-op instead of recreating page state.
            if (blocks_.at(block_index).epoch != block_epoch) {
                return;
            }
            const auto page = programmed_pages_.find(ppn);
            if (page == programmed_pages_.end() ||
                page->second.block_epoch != block_epoch) {
                return;
            }
            mark_programmed(ppn, lpn, owner);
            materialized_ready_by_ppn_[ppn] = std::max(
                materialized_ready_by_ppn_[ppn], commit_ns);
        });
}

void HbfController::schedule_physical_program_commit(
    std::uint64_t ppn,
    double commit_ns) {
    const auto page = programmed_pages_.find(ppn);
    if (page == programmed_pages_.end() || page->second.status != PageStatus::Erased) {
        throw std::runtime_error(
            "HBF physical program commit was scheduled without an erased reservation");
    }
    const auto [pending, inserted] = pending_physical_programs_.emplace(
        ppn,
        PendingPhysicalProgram{.commit_ns = commit_ns});
    if (!inserted) {
        throw std::runtime_error("HBF physical page already has an in-flight program");
    }
    try {
        const auto block_index = static_cast<std::size_t>(
            ppn / config_.device.pages_per_block);
        const auto block_epoch = blocks_.at(block_index).epoch;
        pending->second.commit_sequence = schedule_commit(
            stack_of_block(block_index), commit_ns,
            [this, ppn, block_index, block_epoch]() {
            const auto pending = pending_physical_programs_.find(ppn);
            if (pending == pending_physical_programs_.end()) {
                throw std::runtime_error(
                    "HBF physical program lost its in-flight reservation");
            }
            const auto page_at_commit = programmed_pages_.find(ppn);
            if (blocks_.at(block_index).epoch != block_epoch) {
                // A later physical erase already retired this incarnation;
                // the bytes were programmed and then destroyed, so only the
                // stale state publication is suppressed.
                pending_physical_programs_.erase(pending);
                return;
            }
            if (page_at_commit == programmed_pages_.end() ||
                page_at_commit->second.block_epoch != block_epoch ||
                page_at_commit->second.status != PageStatus::Erased) {
                throw std::runtime_error(
                    "HBF physical program target stopped being erased before commit");
            }
            mark_programmed(ppn, ppn, PageOwner::RawPhysical);
            materialized_ready_by_ppn_[ppn] = std::max(
                materialized_ready_by_ppn_[ppn], pending->second.commit_ns);
            pending_physical_programs_.erase(pending);
        });
    } catch (...) {
        pending_physical_programs_.erase(ppn);
        throw;
    }
}

void HbfController::reserve_physical_program_range(
    std::uint64_t first_ppn,
    std::uint64_t pages) {
    std::unordered_map<std::size_t, std::uint32_t> expected_next;
    std::unordered_map<std::size_t, std::uint32_t> remaining_free;

    // First pass is read-only: reject the whole multi-page request before any
    // block/page allocation state changes.
    for (std::uint64_t i = 0; i < pages; ++i) {
        const auto ppn = first_ppn + i;
        const auto block_index = static_cast<std::size_t>(
            ppn / config_.device.pages_per_block);
        const auto page_index = static_cast<std::uint32_t>(
            ppn % config_.device.pages_per_block);
        const auto& block = blocks_.at(block_index);
        if (block.role != BlockRole::Free && block.role != BlockRole::RawPhysical) {
            throw std::runtime_error(
                block.role == BlockRole::StaticReadOnly ?
                "HBF physical write targets static read-only data" :
                "HBF physical write targets a controller-owned block");
        }
        auto [next_it, next_inserted] = expected_next.emplace(
            block_index, block.next_page);
        auto [free_it, free_inserted] = remaining_free.emplace(
            block_index, block.free_pages);
        (void)next_inserted;
        (void)free_inserted;
        if (page_index != next_it->second) {
            throw std::runtime_error(
                "HBF physical program violates sequential page order");
        }
        if (free_it->second == 0) {
            throw std::runtime_error("HBF physical program selected a full block");
        }
        if (programmed_pages_.contains(ppn) ||
            pending_physical_programs_.contains(ppn)) {
            throw std::runtime_error(
                "HBF physical write targets an allocated or in-flight page");
        }
        next_it->second++;
        free_it->second--;
    }

    for (const auto& entry : expected_next) {
        const auto block_index = entry.first;
        const auto& block = blocks_.at(block_index);
        if (block.role != BlockRole::Free) {
            continue;
        }
        const auto& free_blocks = planes_.at(block_plane_index(block_index)).free_blocks;
        if (std::find(free_blocks.begin(), free_blocks.end(), block_index) ==
            free_blocks.end()) {
            throw std::runtime_error(
                "HBF physical program found a free block missing from its plane pool");
        }
    }

    // Second pass reserves capacity exactly as the FTL allocator does. Page
    // validity remains Erased until each media-completion commit executes.
    for (std::uint64_t i = 0; i < pages; ++i) {
        const auto ppn = first_ppn + i;
        const auto block_index = static_cast<std::size_t>(
            ppn / config_.device.pages_per_block);
        auto& block = blocks_.at(block_index);
        if (block.role == BlockRole::Free) {
            auto& free_blocks = planes_.at(block_plane_index(block_index)).free_blocks;
            const auto found = std::find(free_blocks.begin(), free_blocks.end(), block_index);
            free_blocks.erase(found);
            remove_managed_block_wear(block_index);
            set_block_role(block_index, BlockRole::RawPhysical);
        }
        block.next_page++;
        block.free_pages--;
        mark_gc_headroom_dirty(block_plane_index(block_index));
        block.pending_program_pages++;
        free_pages_--;
        free_pages_per_stack_.at(stack_of_block(block_index))--;
        const auto [page, inserted] = programmed_pages_.emplace(ppn, PageState{});
        if (!inserted) {
            throw std::runtime_error(
                "HBF physical program reservation unexpectedly reused page state");
        }
        page->second.status = PageStatus::Erased;
        page->second.owner = PageOwner::RawPhysical;
        page->second.lpn = 0;
        page->second.block_epoch = block.epoch;
    }
}

void HbfController::prepopulate_raw_physical_page(std::uint64_t byte_address) {
    if (mapping_policy_) throw std::invalid_argument("host mapping owns the media; populate logical pages");
    prepopulate_raw_media_page(byte_address);
}

void HbfController::prepopulate_raw_media_page(std::uint64_t byte_address) {
    if (config_.host.mapping_mode != MappingMode::RawPhysical || last_issue_arrival_ns_ ||
        byte_address % config_.device.page_size_bytes != 0 ||
        byte_address / config_.device.page_size_bytes >= total_pages_) {
        throw std::runtime_error("raw initial image requires an aligned page before timed IO");
    }
    const auto ppn = byte_address / config_.device.page_size_bytes;
    reserve_physical_program_range(ppn, 1);
    mark_programmed(ppn, ppn, PageOwner::RawPhysical);
    stats_.free_pages = free_pages_;
}

void HbfController::prepopulate_logical_pages(const std::vector<std::uint64_t>& lpns) {
    if (mapping_policy_) { mapping_policy_->prepopulate(*this, lpns); return; }
    if (config_.host.mapping_mode == MappingMode::RawPhysical) {
        throw std::runtime_error(
            "HBF raw-physical mode requires an explicit physical initial image");
    }
    if (compact_logical_image_) {
        throw std::runtime_error(
            "HBF cannot add materialized mappings after a compact logical "
            "image");
    }
    std::unordered_set<std::uint64_t> seen;
    for (const auto lpn : lpns) {
        if (!seen.insert(lpn).second || lpn_to_ppn_.find(lpn) != lpn_to_ppn_.end()) {
            continue;
        }
        require_logical_capacity(lpn);
        if (free_pages_ == 0) {
            throw std::runtime_error("HBF cannot prepopulate logical pages: physical capacity exhausted");
        }
        double at_ns = 0.0;
        Breakdown ignored;
        const auto ppn = allocate_free_page(
            at_ns,
            ignored,
            nullptr,
            BlockRole::Data,
            stack_for_lpn(lpn));
        lpn_to_ppn_[lpn] = ppn;
        mark_programmed(ppn, lpn);
        stats_.initial_logical_data_pages = checked_add(
            stats_.initial_logical_data_pages,
            1,
            "HBF materialized initial logical data pages");
        const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
        if (mapping_vpn_to_ppn_.find(mapping_vpn) == mapping_vpn_to_ppn_.end()) {
            const auto mapping_ppn = allocate_free_page(
                at_ns,
                ignored,
                nullptr,
                BlockRole::Mapping,
                stack_for_vpn(mapping_vpn),
                mapping_plane_for_vpn(mapping_vpn));
            mapping_vpn_to_ppn_[mapping_vpn] = mapping_ppn;
            mark_programmed(
                mapping_ppn,
                metadata_lpn(mapping_vpn),
                PageOwner::Mapping);
            stats_.initial_mapping_pages = checked_add(
                stats_.initial_mapping_pages,
                1,
                "HBF materialized initial mapping pages");
        }
    }
    stats_.free_pages = free_pages_;
    refresh_parallel_stats();
}

void HbfController::prepopulate_read_only_logical_page_range(
    std::uint64_t first_lpn,
    std::uint64_t page_count) {
    prepopulate_compact_logical_page_range(
        first_lpn,
        page_count,
        false);
}

void HbfController::prepopulate_mutable_logical_page_range(
    std::uint64_t first_lpn,
    std::uint64_t page_count) {
    prepopulate_compact_logical_page_range(
        first_lpn,
        page_count,
        true);
}

void HbfController::prepopulate_compact_logical_page_range(
    std::uint64_t first_lpn,
    std::uint64_t page_count,
    bool mutable_image) {
    if (mapping_policy_) {
        if (!page_count || first_lpn >= logical_capacity_pages() || page_count > logical_capacity_pages()-first_lpn)
            throw std::invalid_argument("initial mapping range exceeds logical capacity");
        std::vector<std::uint64_t> pages;
        pages.reserve(page_count);
        for (auto p=first_lpn;p<first_lpn+page_count;++p) pages.push_back(p);
        mapping_policy_->prepopulate(*this,pages,!mutable_image);
        return;
    }
    if (config_.host.mapping_mode == MappingMode::RawPhysical) {
        throw std::runtime_error("raw-physical media cannot install an implicit logical image");
    }
    if (page_count == 0) {
        return;
    }
    const auto last_lpn = checked_add(
        first_lpn,
        page_count - 1,
        "HBF prepopulation LPN range");
    require_logical_capacity(last_lpn);
    if (compact_logical_image_ || !lpn_to_ppn_.empty() ||
        !mapping_vpn_to_ppn_.empty()) {
        throw std::runtime_error(
            "HBF compact logical image requires an empty FTL");
    }
    if (!mutable_image &&
        (!programmed_pages_.empty() || free_pages_ != total_pages_)) {
        throw std::runtime_error(
            "HBF compact read-only image must be installed into a fresh "
            "device");
    }
    if (mutable_image) {
        const bool static_only = std::all_of(
            programmed_pages_.begin(),
            programmed_pages_.end(),
            [](const auto& entry) {
                return entry.second.status == PageStatus::StaticReadOnly &&
                    entry.second.owner == PageOwner::StaticReadOnly;
            });
        const bool no_mutable_blocks = std::none_of(
            blocks_.begin(),
            blocks_.end(),
            [](const BlockState& block) {
                return block.role == BlockRole::Data ||
                    block.role == BlockRole::Mapping ||
                    block.role == BlockRole::GC ||
                    block.role == BlockRole::RawPhysical ||
                    block.pending_program_pages != 0 ||
                    block.pending_mapping_publications != 0 ||
                    block.erase_pending;
            });
        if (!static_only || !no_mutable_blocks ||
            !pending_commits_.empty() ||
            !pending_lpn_updates_.empty() ||
            !pending_vpn_updates_.empty() ||
            !dirty_mapping_vpns_.empty()) {
            throw std::runtime_error(
                "HBF compact mutable image must be installed after static "
                "fencing and before mutable FTL work");
        }
    }
    const auto entries_per_mapping_page =
        static_cast<std::uint64_t>(config_.host.mapping_entries_per_page);
    const auto pps = static_cast<std::uint64_t>(planes_per_stack());
    if (pps == 0) {
        throw std::runtime_error(
            "HBF compact logical image requires at least one plane per stack");
    }
    const auto cursor_was_advanced = [](const auto& cursors) {
        return std::any_of(
            cursors.begin(),
            cursors.end(),
            [](std::size_t cursor) { return cursor != 0; });
    };
    if (cursor_was_advanced(next_data_allocation_plane_per_stack_) ||
        cursor_was_advanced(next_mapping_allocation_plane_per_stack_) ||
        cursor_was_advanced(next_gc_allocation_plane_per_stack_)) {
        throw std::runtime_error(
            "HBF compact logical image requires fresh allocation cursors");
    }

    CompactLogicalImage image;
    image.mutable_image = mutable_image;
    image.first_lpn = first_lpn;
    image.page_count = page_count;
    const auto stacks = static_cast<std::uint64_t>(config_.device.stacks);
    const auto first_group =
        (first_lpn / stacks) / entries_per_mapping_page;
    const auto last_group =
        (last_lpn / stacks) / entries_per_mapping_page;
    image.first_vpn = checked_mul(
        first_group,
        stacks,
        "HBF compact first mapping VPN");
    const auto group_count = checked_add(
        last_group - first_group,
        1,
        "HBF compact mapping group count");
    image.vpn_slot_count = checked_mul(
        group_count,
        stacks,
        "HBF compact VPN slot count");
    if (image.vpn_slot_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "HBF compact logical image VPN directory exceeds size_t range");
    }
    image.data_blocks_by_plane.resize(planes_.size());
    image.mapping_ppns.resize(
        static_cast<std::size_t>(image.vpn_slot_count));
    image.vpn_ranges.resize(
        static_cast<std::size_t>(image.vpn_slot_count));
    image.vpn_offsets_by_stack.resize(config_.device.stacks);
    std::vector<std::uint64_t> stack_page_counts(config_.device.stacks, 0);

    for (std::uint64_t vpn_offset = 0;
         vpn_offset < image.vpn_slot_count;
         ++vpn_offset) {
        const auto mapping_vpn = checked_add(
            image.first_vpn, vpn_offset, "HBF compact mapping VPN");
        const auto group = mapping_vpn / stacks;
        const auto stack = stack_for_vpn(mapping_vpn);
        const auto rotation = placement_mix64(group) % stacks;
        const auto lane = static_cast<std::uint64_t>(stack) >= rotation ?
            static_cast<std::uint64_t>(stack) - rotation :
            stacks - (rotation - static_cast<std::uint64_t>(stack));
        const auto group_first_stripe = checked_mul(
            group,
            entries_per_mapping_page,
            "HBF compact mapping-group first stripe");
        const auto group_last_stripe = checked_add(
            group_first_stripe,
            entries_per_mapping_page - 1,
            "HBF compact mapping-group last stripe");
        const auto first_candidate_stripe = [&] {
            if (first_lpn <= lane) {
                return std::uint64_t{0};
            }
            const auto delta = first_lpn - lane;
            return delta / stacks + (delta % stacks == 0 ? 0 : 1);
        }();
        if (last_lpn < lane) {
            continue;
        }
        const auto last_candidate_stripe = (last_lpn - lane) / stacks;
        const auto range_first_stripe = std::max(
            group_first_stripe,
            first_candidate_stripe);
        const auto range_last_stripe = std::min(
            group_last_stripe,
            last_candidate_stripe);
        if (range_first_stripe > range_last_stripe) {
            continue;
        }
        const auto first_entry =
            range_first_stripe - group_first_stripe;
        const auto range_pages = checked_add(
            range_last_stripe - range_first_stripe,
            1,
            "HBF compact mapping-page range");
        if (range_pages > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "HBF compact mapping-page range exceeds supported page count");
        }
        const auto stack_plane_base = stack * planes_per_stack();
        const auto first_data_cursor =
            static_cast<std::uint64_t>(next_data_allocation_plane_per_stack_.at(stack));
        if (first_data_cursor != stack_page_counts.at(stack) % pps) {
            throw std::runtime_error(
                "HBF compact data-allocation cursor lost round-robin order");
        }
        image.vpn_ranges.at(static_cast<std::size_t>(vpn_offset)) =
            CompactLogicalImage::VpnRange{
            .first_entry = first_entry,
            .page_count = range_pages,
            .stack_page_offset = stack_page_counts.at(stack),
        };
        image.vpn_offsets_by_stack.at(stack).push_back(vpn_offset);

        // In the ordinary prepopulation loop the first data page is reserved
        // before the mapping page for this VPN. Preserve that ordering because
        // Data and Mapping may draw their first blocks from the same plane's
        // free-block deque.
        (void)allocate_compact_pages_on_plane(
            stack_plane_base + static_cast<std::size_t>(first_data_cursor),
            BlockRole::Data,
            1,
            &image.data_blocks_by_plane.at(
                stack_plane_base + static_cast<std::size_t>(first_data_cursor)),
            &image.live_data_pages_by_block);
        image.mapping_ppns.at(static_cast<std::size_t>(vpn_offset)) =
            allocate_compact_pages_on_plane(
                mapping_plane_for_vpn(mapping_vpn),
                BlockRole::Mapping,
                1,
                nullptr,
                &image.live_mapping_pages_by_block);
        image.mapping_page_count = checked_add(
            image.mapping_page_count,
            1,
            "HBF compact mapping-page count");

        for (std::size_t local_plane = 0;
             local_plane < planes_per_stack();
             ++local_plane) {
            const auto plane_distance =
                (local_plane + planes_per_stack() -
                    static_cast<std::size_t>(first_data_cursor)) %
                planes_per_stack();
            std::uint64_t plane_pages = range_pages > plane_distance ?
                1 + (range_pages - 1 - plane_distance) / pps : 0;
            if (local_plane == first_data_cursor) {
                --plane_pages;
            }
            if (plane_pages == 0) {
                continue;
            }
            const auto plane = stack_plane_base + local_plane;
            (void)allocate_compact_pages_on_plane(
                plane,
                BlockRole::Data,
                static_cast<std::uint32_t>(plane_pages),
                &image.data_blocks_by_plane.at(plane),
                &image.live_data_pages_by_block);
        }
        stack_page_counts.at(stack) = checked_add(
            stack_page_counts.at(stack),
            range_pages,
            "HBF compact per-stack data pages");
        next_data_allocation_plane_per_stack_.at(stack) =
            static_cast<std::size_t>(stack_page_counts.at(stack) % pps);
    }
    for (std::size_t plane = 0;
         plane < image.data_blocks_by_plane.size();
         ++plane) {
        const auto& assigned = image.data_blocks_by_plane.at(plane);
        for (std::size_t ordinal = 0; ordinal < assigned.size(); ++ordinal) {
            const bool inserted = image.data_block_locations.emplace(
                assigned[ordinal],
                CompactLogicalImage::DataBlockLocation{
                    .plane = plane,
                    .block_ordinal = ordinal,
                }).second;
            if (!inserted) {
                throw std::runtime_error(
                    "HBF compact data block appears in multiple directories");
            }
        }
    }
    for (std::size_t offset = 0;
         offset < image.mapping_ppns.size();
         ++offset) {
        if (!image.mapping_ppns[offset]) {
            continue;
        }
        const auto mapping_vpn = checked_add(
            image.first_vpn,
            offset,
            "HBF compact mapping inverse VPN");
        const bool inserted = image.mapping_vpn_by_ppn.emplace(
            *image.mapping_ppns[offset],
            mapping_vpn).second;
        if (!inserted) {
            throw std::runtime_error(
                "HBF compact mapping PPN is not unique");
        }
    }
    stats_.initial_logical_data_pages = checked_add(
        stats_.initial_logical_data_pages,
        image.page_count,
        "HBF initial logical data pages");
    stats_.initial_mapping_pages = checked_add(
        stats_.initial_mapping_pages,
        image.mapping_page_count,
        "HBF initial mapping pages");
    stats_.compact_initial_logical_data_pages = checked_add(
        stats_.compact_initial_logical_data_pages,
        image.page_count,
        "HBF compact initial logical data pages");
    stats_.compact_initial_mapping_pages = checked_add(
        stats_.compact_initial_mapping_pages,
        image.mapping_page_count,
        "HBF compact initial mapping pages");
    compact_logical_image_ = std::move(image);
    stats_.free_pages = free_pages_;
    refresh_parallel_stats();
}

void HbfController::reserve_static_physical_blocks(
    const std::vector<std::size_t>& block_indices) {
    if (mapping_policy_ && !block_indices.empty())
        throw std::invalid_argument("host mapping owns the media; static reservations are unavailable");
    if (logical_capacity_frozen_ || compact_logical_image_) {
        throw std::runtime_error(
            "HBF cannot reserve static pages after logical use");
    }
    std::unordered_set<std::size_t> touched_blocks(
        block_indices.begin(),
        block_indices.end());
    std::vector<std::uint8_t> reserve_mask(blocks_.size(), 0);
    std::vector<std::size_t> reserve_count_by_plane(planes_.size(), 0);
    std::size_t blocks_to_reserve = 0;
    for (const auto block_index : touched_blocks) {
        if (block_index >= blocks_.size()) {
            throw std::runtime_error(
                "HBF static-data block index is out of range");
        }
    }

    // Validate the entire request before changing any pool, role, bitmap, or
    // capacity counter. A mixed legal/illegal list must fail closed rather
    // than leave whichever unordered-set element happened to be visited first
    // as a partially reserved static block.
    for (const auto block_index : touched_blocks) {
        const auto& block = blocks_.at(block_index);
        if (block.role == BlockRole::StaticReadOnly) {
            continue;
        }
        if (block.role != BlockRole::Free || block.free_pages != config_.device.pages_per_block ||
            block.valid_pages != 0 || block.invalid_pages != 0 || block.next_page != 0 ||
            block.pending_program_pages != 0 ||
            block.pending_mapping_publications != 0) {
            throw std::runtime_error(
                "HBF static-data reservation intersects live FTL state at block " +
                std::to_string(block_index));
        }
        reserve_mask.at(block_index) = 1;
        ++blocks_to_reserve;
        ++reserve_count_by_plane.at(block_plane_index(block_index));
    }
    // Dense extents can cover hundreds of thousands of blocks. Validate pool
    // membership in one linear scan instead of performing one O(blocks-per-
    // plane) find for every reserved block.
    std::size_t pool_matches = 0;
    for (const auto& plane : planes_) {
        for (const auto block_index : plane.free_blocks) {
            pool_matches += reserve_mask.at(block_index) != 0 ? 1 : 0;
        }
    }
    if (pool_matches != blocks_to_reserve) {
        throw std::runtime_error(
            "HBF free block is missing from its plane pool");
    }

    // Reserve complete NAND blocks. Page-granular fencing would let the FTL
    // program around immutable wordlines and later collide with them, which is
    // not a valid sequential-program model.
    for (std::size_t plane_index = 0;
         plane_index < planes_.size();
         ++plane_index) {
        if (reserve_count_by_plane.at(plane_index) == 0) {
            continue;
        }
        auto& free_blocks = planes_.at(plane_index).free_blocks;
        const auto first_removed = std::remove_if(
            free_blocks.begin(),
            free_blocks.end(),
            [&](std::size_t block_index) {
                return reserve_mask.at(block_index) != 0;
            });
        const auto removed = static_cast<std::size_t>(
            std::distance(first_removed, free_blocks.end()));
        if (removed != reserve_count_by_plane.at(plane_index)) {
            throw std::runtime_error(
                "HBF static-data free-pool removal count mismatch");
        }
        mark_gc_headroom_dirty(plane_index);
        free_blocks.erase(first_removed, free_blocks.end());
    }
    for (const auto block_index : touched_blocks) {
        auto& block = blocks_.at(block_index);
        if (block.role == BlockRole::StaticReadOnly) {
            continue;
        }
        remove_managed_block_wear(block_index);
        set_block_role(block_index, BlockRole::StaticReadOnly);
        block.free_pages = 0;
        block.next_page = config_.device.pages_per_block;
        free_pages_ -= config_.device.pages_per_block;
        free_pages_per_stack_.at(stack_of_block(block_index)) -= config_.device.pages_per_block;
        stats_.static_reserved_pages += config_.device.pages_per_block;
    }
    stats_.free_pages = free_pages_;
}

void HbfController::reserve_static_physical_pages(
    const std::vector<std::uint64_t>& ppns) {
    std::unordered_set<std::uint64_t> unique_pages;
    std::vector<std::size_t> touched_blocks;
    touched_blocks.reserve(ppns.size());
    for (const auto ppn : ppns) {
        if (ppn >= total_pages_) {
            throw std::runtime_error("HBF static-data PPN is out of range");
        }
        if (unique_pages.insert(ppn).second) {
            touched_blocks.push_back(
                static_cast<std::size_t>(ppn / config_.device.pages_per_block));
        }
    }
    // Page conflicts must be checked before the block helper mutates any
    // ownership state.
    for (const auto ppn : unique_pages) {
        const auto page = programmed_pages_.find(ppn);
        if (page != programmed_pages_.end() &&
            page->second.status != PageStatus::Erased &&
            page->second.status != PageStatus::StaticReadOnly) {
            throw std::runtime_error(
                "HBF static-data reservation intersects a programmed page");
        }
    }
    reserve_static_physical_blocks(touched_blocks);
    for (const auto ppn : unique_pages) {
        auto& page = programmed_pages_[ppn];
        if (page.status == PageStatus::Erased) {
            page.status = PageStatus::StaticReadOnly;
            page.owner = PageOwner::StaticReadOnly;
            page.lpn = ppn;
            auto& block = blocks_.at(static_cast<std::size_t>(ppn / config_.device.pages_per_block));
            page.block_epoch = block.epoch;
            block.valid_pages++;
            block.set_valid(static_cast<std::uint32_t>(ppn % config_.device.pages_per_block), config_.device.pages_per_block);
        }
    }
    stats_.free_pages = free_pages_;
}

void HbfController::reserve_static_physical_block_indices(
    const std::vector<std::size_t>& block_indices) {
    reserve_static_physical_blocks(block_indices);
}

void HbfController::reserve_static_physical_block_extent(
    std::uint32_t first_block,
    std::uint32_t block_count) {
    const auto end_block = static_cast<std::uint64_t>(first_block) +
        static_cast<std::uint64_t>(block_count);
    if (end_block > config_.device.blocks_per_plane) {
        throw std::runtime_error(
            "HBF static-data block extent exceeds blocks-per-plane geometry");
    }
    if (block_count == 0) {
        return;
    }
    std::vector<std::size_t> block_indices;
    if (planes_.size() >
        std::numeric_limits<std::size_t>::max() / block_count) {
        throw std::runtime_error(
            "HBF static-data block extent size exceeds size_t range");
    }
    block_indices.reserve(
        planes_.size() * static_cast<std::size_t>(block_count));
    for (std::size_t plane = 0; plane < planes_.size(); ++plane) {
        const auto plane_base =
            plane * static_cast<std::size_t>(config_.device.blocks_per_plane);
        for (std::uint64_t block = first_block; block < end_block; ++block) {
            block_indices.push_back(
                plane_base + static_cast<std::size_t>(block));
        }
    }
    reserve_static_physical_blocks(block_indices);
}

void HbfController::reserve_raw_physical_block_extent(
    std::uint32_t first_block,
    std::uint32_t block_count) {
    if (mapping_policy_ && block_count)
        throw std::invalid_argument("host mapping owns the media; raw reservations are unavailable");
    const auto end_block = static_cast<std::uint64_t>(first_block) +
        static_cast<std::uint64_t>(block_count);
    if (end_block > config_.device.blocks_per_plane) {
        throw std::runtime_error(
            "HBF raw physical block extent exceeds blocks-per-plane geometry");
    }
    if (block_count == 0) {
        return;
    }
    if (logical_capacity_frozen_ || compact_logical_image_) {
        throw std::runtime_error(
            "HBF cannot reserve raw physical blocks after logical use");
    }

    std::vector<std::uint8_t> reserve_mask(blocks_.size(), 0);
    std::vector<std::size_t> reserve_count_by_plane(planes_.size(), 0);
    std::size_t blocks_to_reserve = 0;
    for (std::size_t plane = 0; plane < planes_.size(); ++plane) {
        const auto plane_base =
            plane * static_cast<std::size_t>(config_.device.blocks_per_plane);
        for (std::uint64_t local = first_block; local < end_block; ++local) {
            const auto block_index =
                plane_base + static_cast<std::size_t>(local);
            const auto& block = blocks_.at(block_index);
            if (block.role == BlockRole::RawPhysical) {
                continue;
            }
            if (block.role != BlockRole::Free ||
                block.free_pages != config_.device.pages_per_block ||
                block.valid_pages != 0 || block.invalid_pages != 0 ||
                block.next_page != 0 || block.pending_program_pages != 0 ||
                block.pending_mapping_publications != 0 ||
                block.erase_pending) {
                throw std::runtime_error(
                    "HBF raw physical reservation intersects owned media at block " +
                    std::to_string(block_index));
            }
            reserve_mask.at(block_index) = 1;
            ++reserve_count_by_plane.at(plane);
            ++blocks_to_reserve;
        }
    }

    std::size_t pool_matches = 0;
    for (const auto& plane : planes_) {
        for (const auto block_index : plane.free_blocks) {
            pool_matches += reserve_mask.at(block_index) != 0 ? 1 : 0;
        }
    }
    if (pool_matches != blocks_to_reserve) {
        throw std::runtime_error(
            "HBF raw physical free block is missing from its plane pool");
    }
    for (std::size_t plane = 0; plane < planes_.size(); ++plane) {
        if (reserve_count_by_plane.at(plane) == 0) {
            continue;
        }
        auto& free_blocks = planes_.at(plane).free_blocks;
        const auto first_removed = std::remove_if(
            free_blocks.begin(),
            free_blocks.end(),
            [&](std::size_t block_index) {
                return reserve_mask.at(block_index) != 0;
            });
        const auto removed = static_cast<std::size_t>(
            std::distance(first_removed, free_blocks.end()));
        if (removed != reserve_count_by_plane.at(plane)) {
            throw std::runtime_error(
                "HBF raw physical free-pool removal count mismatch");
        }
        mark_gc_headroom_dirty(plane);
        free_blocks.erase(first_removed, free_blocks.end());
    }
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        if (reserve_mask.at(block_index) == 0) {
            continue;
        }
        remove_managed_block_wear(block_index);
        set_block_role(block_index, BlockRole::RawPhysical);
        stats_.raw_reserved_pages = checked_add(
            stats_.raw_reserved_pages,
            config_.device.pages_per_block,
            "HBF raw reserved-page accounting");
    }
}

void HbfController::validate_raw_physical_block_extent(
    std::uint32_t first_block,
    std::uint32_t block_count) const {
    const auto end_block = static_cast<std::uint64_t>(first_block) +
        static_cast<std::uint64_t>(block_count);
    if (block_count == 0 || end_block > config_.device.blocks_per_plane) {
        throw std::runtime_error(
            "HBF restored raw extent declaration is empty or out of range");
    }
    std::uint64_t raw_blocks = 0;
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        const auto local = static_cast<std::uint64_t>(
            block_index % config_.device.blocks_per_plane);
        const bool expected = local >= first_block && local < end_block;
        const bool actual = blocks_.at(block_index).role == BlockRole::RawPhysical;
        if (expected != actual) {
            throw std::runtime_error(
                "HBF restored raw physical ownership does not match the declared extent");
        }
        raw_blocks = checked_add(
            raw_blocks, actual ? 1 : 0, "HBF restored raw-block accounting");
    }
    const auto expected_blocks = checked_mul(
        static_cast<std::uint64_t>(planes_.size()),
        block_count,
        "HBF restored raw extent block count");
    if (raw_blocks != expected_blocks) {
        throw std::runtime_error(
            "HBF restored raw physical block count diverged");
    }
}

PhysicalCompletion HbfController::drain_pending(
    std::string id,
    double arrival_ns,
    TraceConfig trace) {
    if (mapping_policy_) return mapping_policy_->checkpoint(*this, std::move(id), arrival_ns, trace);
    PhysicalCompletion out;
    out.id = std::move(id);
    out.tier = Tier::HBF;
    out.op = Op::Write;
    out.arrival_ns = arrival_ns;
    out.start_ns = arrival_ns;
    out.logical_bytes = 0;
    out.resource_path = "logic/write_buffer+mapping_table";
    auto* trace_spans = trace_spans_enabled(trace) ? &out.spans : nullptr;
    const auto physical_bytes_before = stats_.physical_read_bytes + stats_.physical_write_bytes;

    if (!std::isfinite(arrival_ns) || arrival_ns < 0.0) {
        throw std::runtime_error("HBF drain arrival must be finite and non-negative");
    }
    if (last_issue_arrival_ns_ && arrival_ns < *last_issue_arrival_ns_) {
        throw std::runtime_error("HBF drain cannot precede the last issue/drain arrival");
    }
    // drain_pending is a top-level causal barrier. Recording its arrival also
    // prevents a later issue from travelling behind calendar history that the
    // drain is now allowed to reclaim.
    last_issue_arrival_ns_ = arrival_ns;
    reservation_causal_watermark_ns_ = arrival_ns;
    prune_expired_state(arrival_ns);
    seed_media_image();
    double causal_ready_ns = 0.0;
    for (const auto ready_ns : causal_state_ready_by_stack_) {
        causal_ready_ns = std::max(causal_ready_ns, ready_ns);
    }
    double at_ns = std::max({arrival_ns, background_finish_ns_, causal_ready_ns});
    apply_commits_through(at_ns);
    std::fill(
        state_observation_by_stack_.begin(),
        state_observation_by_stack_.end(),
        at_ns);
    advance_administrative_frontier(at_ns);
    flush_all_write_buffer_entries(at_ns, out.breakdown, trace_spans);
    advance_administrative_frontier(at_ns);
    complete_active_relocations(at_ns, out.breakdown, trace_spans);
    advance_administrative_frontier(at_ns);
    flush_all_dirty_mapping_pages(at_ns, out.breakdown, trace_spans);
    at_ns = std::max(at_ns, background_finish_ns_);
    if (!pending_commits_.empty()) {
        at_ns = std::max(at_ns, pending_commits_.rbegin()->first.first);
    }
    apply_commits_through(at_ns);
    out.finish_ns = at_ns;
    advance_administrative_frontier(at_ns);
    for (std::size_t stack = 0; stack < config_.device.stacks; ++stack) {
        causal_state_ready_by_stack_[stack] = std::max(
            causal_state_ready_by_stack_[stack], out.finish_ns);
        state_observation_by_stack_[stack] = std::max(
            state_observation_by_stack_[stack], out.finish_ns);
    }
    out.note = out.finish_ns > out.arrival_ns ? "drained-pending-hbf-state" : "no-pending-hbf-state";
    out.physical_bytes = stats_.physical_read_bytes + stats_.physical_write_bytes -
        physical_bytes_before;

    stats_.free_pages = free_pages_;
    stats_.stage_work += out.breakdown;
    stats_.finish_ns = std::max({stats_.finish_ns, out.finish_ns, background_finish_ns_});
    return out;
}

std::vector<HbfBlockProfileBin> HbfController::block_profile(
    std::size_t bin_count) const {
    if (bin_count == 0 || blocks_.size() % bin_count != 0) {
        throw std::runtime_error(
            "HBF block profile bin count must divide the block count");
    }
    const auto blocks_per_bin = blocks_.size() / bin_count;
    std::vector<HbfBlockProfileBin> profile(bin_count);
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        const auto& block = blocks_[block_index];
        auto& bin = profile[block_index / blocks_per_bin];
        bin.blocks++;
        bin.erase_count_sum += block.erase_count;
        bin.min_erase_count = std::min(bin.min_erase_count, block.erase_count);
        bin.max_erase_count = std::max(bin.max_erase_count, block.erase_count);
        switch (block.role) {
        case BlockRole::StaticReadOnly:
            bin.static_blocks++;
            bin.valid_pages += config_.device.pages_per_block;
            continue;
        case BlockRole::Free:
            bin.free_blocks++;
            break;
        case BlockRole::Data:
            bin.data_blocks++;
            break;
        case BlockRole::GC:
            bin.gc_blocks++;
            break;
        case BlockRole::Mapping:
            bin.mapping_blocks++;
            break;
        case BlockRole::RawPhysical:
            bin.raw_physical_blocks++;
            break;
        }
        bin.valid_pages += block.valid_pages;
        bin.invalid_pages += block.invalid_pages;
        bin.free_pages += block.free_pages;
        bin.pending_pages += block.pending_program_pages;
    }
    return profile;
}

std::vector<std::uint32_t> HbfController::block_erase_counts() const {
    std::vector<std::uint32_t> counts;
    counts.reserve(blocks_.size());
    for (std::size_t block_index = 0;
         block_index < blocks_.size();
         ++block_index) {
        const auto& block = blocks_[block_index];
        if (block.role == BlockRole::StaticReadOnly) {
            continue;
        }
        counts.push_back(block.erase_count);
    }
    return counts;
}

HbfQuiescenceStats HbfController::quiescence_stats() const {
    HbfQuiescenceStats result;
    result.dirty_mapping_pages = dirty_mapping_vpns_.size();
    if (mapping_policy_ && mapping_policy_->dirty()) ++result.dirty_mapping_pages;
    if (mapping_policy_) result.write_buffer_entries=mapping_policy_->buffered_pages();
    for (const auto& [_, events] : pending_dirty_mapping_events_) {
        result.pending_dirty_mapping_events = checked_add(
            result.pending_dirty_mapping_events,
            events.size(),
            "HBF pending dirty-mapping event count");
    }
    for (const auto& [_, updates] : pending_lpn_updates_) {
        result.pending_lpn_updates = checked_add(
            result.pending_lpn_updates,
            updates.size(),
            "HBF pending LPN update count");
    }
    for (const auto& [_, updates] : pending_vpn_updates_) {
        result.pending_vpn_updates = checked_add(
            result.pending_vpn_updates,
            updates.size(),
            "HBF pending VPN update count");
    }
    result.pending_commits = pending_commits_.size();
    for (const auto& buffer : write_buffer_by_stack_) {
        result.write_buffer_entries = checked_add(
            result.write_buffer_entries,
            buffer.size(),
            "HBF write-buffer entry count");
    }
    for (const auto& [_, generations] : inflight_buffered_writes_) {
        result.inflight_buffered_generations = checked_add(
            result.inflight_buffered_generations,
            generations.size(),
            "HBF inflight buffered-generation count");
    }
    result.pending_physical_programs = pending_physical_programs_.size();
    result.pending_block_transitions = pending_block_transitions_.size();
    return result;
}

void HbfController::materialize_committed_state_through(
    double completed_frontier_ns) {
    if (!std::isfinite(completed_frontier_ns) ||
        completed_frontier_ns < 0.0) {
        throw std::runtime_error(
            "HBF completed-state frontier must be finite and non-negative");
    }
    apply_commits_through(completed_frontier_ns);
}

HbfAuditSnapshot HbfController::audit_snapshot() const {
    HbfAuditSnapshot snapshot;
    snapshot.free_pages = free_pages_;
    snapshot.free_pages_per_stack = free_pages_per_stack_;
    snapshot.data_allocation_cursors.assign(
        next_data_allocation_plane_per_stack_.begin(),
        next_data_allocation_plane_per_stack_.end());
    snapshot.mapping_allocation_cursors.assign(
        next_mapping_allocation_plane_per_stack_.begin(),
        next_mapping_allocation_plane_per_stack_.end());
    snapshot.gc_allocation_cursors.assign(
        next_gc_allocation_plane_per_stack_.begin(),
        next_gc_allocation_plane_per_stack_.end());

    snapshot.logical_mappings.reserve(lpn_to_ppn_.size());
    for (const auto& [lpn, ppn] : lpn_to_ppn_) {
        snapshot.logical_mappings.push_back({.key = lpn, .ppn = ppn});
    }
    std::sort(
        snapshot.logical_mappings.begin(),
        snapshot.logical_mappings.end(),
        [](const HbfAuditMapping& lhs, const HbfAuditMapping& rhs) {
            return std::tie(lhs.key, lhs.ppn) < std::tie(rhs.key, rhs.ppn);
        });

    snapshot.mapping_pages.reserve(mapping_vpn_to_ppn_.size());
    for (const auto& [vpn, ppn] : mapping_vpn_to_ppn_) {
        snapshot.mapping_pages.push_back({.key = vpn, .ppn = ppn});
    }
    std::sort(
        snapshot.mapping_pages.begin(),
        snapshot.mapping_pages.end(),
        [](const HbfAuditMapping& lhs, const HbfAuditMapping& rhs) {
            return std::tie(lhs.key, lhs.ppn) < std::tie(rhs.key, rhs.ppn);
        });

    const auto page_status_name = [](PageStatus status) -> std::string {
        switch (status) {
        case PageStatus::Erased:
            return "erased";
        case PageStatus::StaticReadOnly:
            return "static_read_only";
        case PageStatus::Valid:
            return "valid";
        }
        throw std::runtime_error("unknown HBF page status");
    };
    const auto page_owner_name = [](PageOwner owner) -> std::string {
        switch (owner) {
        case PageOwner::Unassigned:
            return "unassigned";
        case PageOwner::Logical:
            return "logical";
        case PageOwner::Mapping:
            return "mapping";
        case PageOwner::RawPhysical:
            return "raw_physical";
        case PageOwner::StaticReadOnly:
            return "static_read_only";
        }
        throw std::runtime_error("unknown HBF page owner");
    };
    snapshot.materialized_pages.reserve(programmed_pages_.size());
    for (const auto& [ppn, page] : programmed_pages_) {
        snapshot.materialized_pages.push_back({
            .ppn = ppn,
            .status = page_status_name(page.status),
            .owner = page_owner_name(page.owner),
            .logical_key = page.lpn,
            .block_epoch = page.block_epoch,
        });
    }
    std::sort(
        snapshot.materialized_pages.begin(),
        snapshot.materialized_pages.end(),
        [](const HbfAuditPage& lhs, const HbfAuditPage& rhs) {
            return lhs.ppn < rhs.ppn;
        });

    const auto block_role_name = [](BlockRole role) -> std::string {
        switch (role) {
        case BlockRole::Free:
            return "free";
        case BlockRole::StaticReadOnly:
            return "static_read_only";
        case BlockRole::RawPhysical:
            return "raw_physical";
        case BlockRole::Data:
            return "data";
        case BlockRole::Mapping:
            return "mapping";
        case BlockRole::GC:
            return "gc";
        }
        throw std::runtime_error("unknown HBF block role");
    };
    snapshot.blocks.reserve(blocks_.size());
    for (std::size_t index = 0; index < blocks_.size(); ++index) {
        const auto& block = blocks_[index];
        snapshot.blocks.push_back({
            .block = index,
            .role = block_role_name(block.role),
            .valid_pages = block.valid_pages,
            .invalid_pages = block.invalid_pages,
            .free_pages = block.free_pages,
            .next_page = block.next_page,
            .erase_count = block.erase_count,
            .pending_program_pages = block.pending_program_pages,
            .pending_mapping_publications =
                block.pending_mapping_publications,
            .epoch = block.epoch,
            .erase_pending = block.erase_pending ||
                pending_block_transitions_.contains(index),
            .initial_raw_pages = block.initial_raw_pages,
        });
    }

    snapshot.dirty_mapping_vpns.assign(
        dirty_mapping_vpns_.begin(), dirty_mapping_vpns_.end());
    std::sort(
        snapshot.dirty_mapping_vpns.begin(),
        snapshot.dirty_mapping_vpns.end());
    for (const auto& [_, events] : pending_dirty_mapping_events_) {
        snapshot.pending_dirty_mapping_events += events.size();
    }
    for (const auto& [_, updates] : pending_lpn_updates_) {
        snapshot.pending_lpn_updates += updates.size();
    }
    for (const auto& [_, updates] : pending_vpn_updates_) {
        snapshot.pending_vpn_updates += updates.size();
    }
    snapshot.pending_commits = pending_commits_.size();
    for (const auto& buffer : write_buffer_by_stack_) {
        snapshot.write_buffer_entries += buffer.size();
    }
    for (const auto& [_, generations] : inflight_buffered_writes_) {
        snapshot.inflight_buffered_generations += generations.size();
    }
    snapshot.pending_physical_programs = pending_physical_programs_.size();
    snapshot.pending_block_transitions = pending_block_transitions_.size();
    return snapshot;
}

HbfPersistentImage HbfController::persistent_image() const {
    const auto quiescence = quiescence_stats();
    if (!quiescence.quiescent()) {
        throw std::runtime_error(
            "HBF persistent image requires quiescent media and FTL state");
    }
    // The image format has no erased-block state: a free block restores as
    // not erased. Refuse rather than silently turn pre-erased blocks dirty.
    if (media_->unconsumed_erased_blocks()) {
        throw std::runtime_error(
            "HBF persistent image cannot carry pre-erased free blocks "
            "(hbf-initial-free-blocks-erased); export before the first command "
            "or disable the precondition");
    }
    HbfPersistentImage image{
        .version = 7,
        .zone_size_blocks = config_.host.zone_size_blocks,
        .channels_per_stack = config_.device.channels_per_stack,
        .zone_managed = zone_managed_,
        .zone_remapping = media_->zone_mapping(),
        .stacks = config_.device.stacks,
        .planes = planes_.size(),
        .blocks_per_plane = config_.device.blocks_per_plane,
        .pages_per_block = config_.device.pages_per_block,
        .page_size_bytes = config_.device.page_size_bytes,
        .mapping_entries_per_page = config_.host.mapping_entries_per_page,
        .state = audit_snapshot(),
    };
    image.plane_state.reserve(planes_.size());
    for (const auto& plane : planes_) {
        HbfPersistentPlane persistent;
        persistent.free_blocks.reserve(plane.free_blocks.size());
        for (const auto block : plane.free_blocks) {
            persistent.free_blocks.push_back(block);
        }
        if (plane.active_data_block) {
            persistent.active_data_block = *plane.active_data_block;
        }
        if (plane.active_mapping_block) {
            persistent.active_mapping_block = *plane.active_mapping_block;
        }
        if (plane.active_gc_block) {
            persistent.active_gc_block = *plane.active_gc_block;
        }
        image.plane_state.push_back(std::move(persistent));
    }
    if (compact_logical_image_) {
        const auto& compact = *compact_logical_image_;
        HbfPersistentCompactImage persistent{
            .mutable_image = compact.mutable_image,
            .first_lpn = compact.first_lpn,
            .page_count = compact.page_count,
            .first_vpn = compact.first_vpn,
            .vpn_slot_count = compact.vpn_slot_count,
            .mapping_page_count = compact.mapping_page_count,
            .data_blocks_by_plane = compact.data_blocks_by_plane,
            .mapping_ppns = compact.mapping_ppns,
        };
        persistent.vpn_ranges.reserve(compact.vpn_ranges.size());
        for (const auto& range : compact.vpn_ranges) {
            persistent.vpn_ranges.push_back({
                .first_entry = range.first_entry,
                .page_count = range.page_count,
                .stack_page_offset = range.stack_page_offset,
            });
        }
        const auto sorted_live_blocks = [](const auto& source) {
            std::vector<HbfPersistentCompactLiveBlock> result;
            result.reserve(source.size());
            for (const auto& [block, pages] : source) {
                result.push_back({.block = block, .live_pages = pages});
            }
            std::sort(
                result.begin(), result.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.block < rhs.block;
                });
            return result;
        };
        persistent.live_data_pages_by_block =
            sorted_live_blocks(compact.live_data_pages_by_block);
        persistent.live_mapping_pages_by_block =
            sorted_live_blocks(compact.live_mapping_pages_by_block);
        persistent.retired_lpns = compact.retired_lpns;
        persistent.retired_mapping_vpns = compact.retired_mapping_vpns;
        image.compact_image = std::move(persistent);
    }
    if (mapping_policy_) image.host_mapping_state = mapping_policy_->snapshot();
    return image;
}

void HbfController::restore_persistent_image(const HbfPersistentImage& image) {
    // Free blocks of a restored image may have been reclaimed without an
    // erase; only a freshly installed image can declare them pre-erased.
    if (config_.host.initial_free_blocks_erased)
        throw std::invalid_argument(
            "hbf-initial-free-blocks-erased applies to a freshly installed image, not a restored one");
    if (static_cast<bool>(mapping_policy_) != !image.host_mapping_state.empty())
        throw std::invalid_argument("persistent image mapping organization does not match this controller");
    if (config_.host.mapping_mode == MappingMode::RawPhysical && image.compact_image) {
        throw std::runtime_error("raw-physical media cannot restore an implicit logical image");
    }
    media_->validate_zone_mapping(image.zone_remapping);
    if (!image.zone_managed && !image.zone_remapping.empty())
        throw std::runtime_error("HBF zone map requires host zone ownership");
    const bool fresh_blocks = std::all_of(
        blocks_.begin(),
        blocks_.end(),
        [this](const BlockState& block) {
            return block.role == BlockRole::Free &&
                block.valid_pages == 0 && block.invalid_pages == 0 &&
                block.free_pages == config_.device.pages_per_block &&
                block.next_page == 0 && block.erase_count == 0 &&
                block.pending_program_pages == 0 &&
                block.pending_mapping_publications == 0 &&
                block.epoch == 0 && !block.erase_pending;
        });
    if (!fresh_blocks || compact_logical_image_ || !lpn_to_ppn_.empty() ||
        !mapping_vpn_to_ppn_.empty() || !programmed_pages_.empty() ||
        free_pages_ != total_pages_ || restored_block_erases_ != 0 ||
        last_issue_arrival_ns_) {
        throw std::runtime_error(
            "HBF persistent image restore requires a fresh device");
    }
    if (image.version != 7 || image.zone_size_blocks != config_.host.zone_size_blocks ||
        image.channels_per_stack != config_.device.channels_per_stack || image.stacks != config_.device.stacks ||
        image.planes != planes_.size() ||
        image.blocks_per_plane != config_.device.blocks_per_plane ||
        image.pages_per_block != config_.device.pages_per_block ||
        image.page_size_bytes != config_.device.page_size_bytes ||
        image.mapping_entries_per_page != config_.host.mapping_entries_per_page ||
        image.state.blocks.size() != blocks_.size() ||
        image.plane_state.size() != planes_.size() ||
        image.state.free_pages_per_stack.size() != config_.device.stacks ||
        image.state.data_allocation_cursors.size() != config_.device.stacks ||
        image.state.mapping_allocation_cursors.size() != config_.device.stacks ||
        image.state.gc_allocation_cursors.size() != config_.device.stacks) {
        throw std::runtime_error(
            "HBF persistent image version or geometry does not match the device");
    }
    if (!image.state.quiescent()) {
        throw std::runtime_error(
            "HBF persistent image contains volatile or pending state");
    }

    const auto block_role = [](const std::string& name) {
        if (name == "free") return BlockRole::Free;
        if (name == "static_read_only") return BlockRole::StaticReadOnly;
        if (name == "raw_physical") return BlockRole::RawPhysical;
        if (name == "data") return BlockRole::Data;
        if (name == "mapping") return BlockRole::Mapping;
        if (name == "gc") return BlockRole::GC;
        throw std::runtime_error(
            "HBF persistent image contains an unknown block role");
    };
    const auto page_status = [](const std::string& name) {
        if (name == "static_read_only") return PageStatus::StaticReadOnly;
        if (name == "valid") return PageStatus::Valid;
        if (name == "erased") return PageStatus::Erased;
        throw std::runtime_error(
            "HBF persistent image contains an unknown page status");
    };
    const auto page_owner = [](const std::string& name) {
        if (name == "unassigned") return PageOwner::Unassigned;
        if (name == "logical") return PageOwner::Logical;
        if (name == "mapping") return PageOwner::Mapping;
        if (name == "raw_physical") return PageOwner::RawPhysical;
        if (name == "static_read_only") return PageOwner::StaticReadOnly;
        throw std::runtime_error(
            "HBF persistent image contains an unknown page owner");
    };

    reset_gc_headroom_cache();
    for (std::size_t index = 0; index < image.state.blocks.size(); ++index) {
        const auto& source = image.state.blocks[index];
        if (source.block != index || source.pending_program_pages != 0 ||
            source.pending_mapping_publications != 0 || source.erase_pending) {
            throw std::runtime_error(
                "HBF persistent image contains non-quiescent block state");
        }
        auto& destination = blocks_[index];
        destination = BlockState{};
        destination.role = block_role(source.role);
        destination.valid_pages = source.valid_pages;
        destination.invalid_pages = source.invalid_pages;
        destination.free_pages = source.free_pages;
        destination.next_page = source.next_page;
        destination.erase_count = source.erase_count;
        destination.epoch = source.epoch;
        destination.initial_raw_pages = source.initial_raw_pages;
        if (source.initial_raw_pages) {
            if (destination.role != BlockRole::RawPhysical ||
                source.initial_raw_pages != config_.device.pages_per_block ||
                source.valid_pages != source.initial_raw_pages)
                throw std::runtime_error("invalid compact raw initial block in HBF image");
            destination.set_valid_range(0, source.initial_raw_pages, config_.device.pages_per_block);
        }
    }
    rebuild_active_plane_counts();

    lpn_to_ppn_.reserve(image.state.logical_mappings.size());
    for (const auto& mapping : image.state.logical_mappings) {
        if (!lpn_to_ppn_.emplace(mapping.key, mapping.ppn).second) {
            throw std::runtime_error(
                "HBF persistent image duplicates an L2P key");
        }
    }
    mapping_vpn_to_ppn_.reserve(image.state.mapping_pages.size());
    for (const auto& mapping : image.state.mapping_pages) {
        if (!mapping_vpn_to_ppn_.emplace(mapping.key, mapping.ppn).second) {
            throw std::runtime_error(
                "HBF persistent image duplicates a mapping-page key");
        }
    }
    programmed_pages_.reserve(image.state.materialized_pages.size());
    for (const auto& source : image.state.materialized_pages) {
        if (source.ppn >= total_pages_) {
            throw std::runtime_error(
                "HBF persistent image contains an out-of-range page");
        }
        const auto status = page_status(source.status);
        if (status == PageStatus::Erased) {
            throw std::runtime_error(
                "HBF quiescent persistent image contains an erased pending page");
        }
        const auto owner = page_owner(source.owner);
        const auto block_index = static_cast<std::size_t>(
            source.ppn / config_.device.pages_per_block);
        const auto page_index = static_cast<std::uint32_t>(
            source.ppn % config_.device.pages_per_block);
        if (source.block_epoch != blocks_[block_index].epoch ||
            !programmed_pages_.emplace(
                source.ppn,
                PageState{
                    .status = status,
                    .owner = owner,
                    .lpn = source.logical_key,
                    .block_epoch = source.block_epoch,
                }).second) {
            throw std::runtime_error(
                "HBF persistent image duplicates or mis-epochs a page");
        }
        if (status == PageStatus::Valid ||
            status == PageStatus::StaticReadOnly) {
            blocks_[block_index].set_valid(page_index, config_.device.pages_per_block);
        }
    }

    const auto pps = planes_per_stack();
    const auto restore_cursors = [pps](
                                     const std::vector<std::uint64_t>& source,
                                     std::vector<std::size_t>& destination) {
        destination.clear();
        destination.reserve(source.size());
        for (const auto cursor : source) {
            if (cursor >= pps) {
                throw std::runtime_error(
                    "HBF persistent image allocation cursor is out of range");
            }
            destination.push_back(static_cast<std::size_t>(cursor));
        }
    };
    restore_cursors(
        image.state.data_allocation_cursors,
        next_data_allocation_plane_per_stack_);
    restore_cursors(
        image.state.mapping_allocation_cursors,
        next_mapping_allocation_plane_per_stack_);
    restore_cursors(
        image.state.gc_allocation_cursors,
        next_gc_allocation_plane_per_stack_);

    for (std::size_t plane_index = 0;
         plane_index < planes_.size();
         ++plane_index) {
        auto& plane = planes_[plane_index];
        const auto& source = image.plane_state[plane_index];
        plane.free_blocks.clear();
        for (const auto block : source.free_blocks) {
            if (block >= blocks_.size() ||
                block_plane_index(static_cast<std::size_t>(block)) !=
                    plane_index) {
                throw std::runtime_error(
                    "HBF persistent image free-block order is out of plane");
            }
            plane.free_blocks.push_back(static_cast<std::size_t>(block));
        }
        const auto restore_active = [this, plane_index](
                                        std::optional<std::uint64_t> block,
                                        BlockRole expected) ->
                                        std::optional<std::size_t> {
            if (!block) return std::nullopt;
            if (*block >= blocks_.size() ||
                block_plane_index(static_cast<std::size_t>(*block)) !=
                    plane_index) {
                throw std::runtime_error(
                    "HBF persistent image active block is out of plane");
            }
            const auto& state = blocks_[static_cast<std::size_t>(*block)];
            if (state.role != expected || state.free_pages == 0) {
                throw std::runtime_error(
                    "HBF persistent image active block has the wrong role");
            }
            return static_cast<std::size_t>(*block);
        };
        plane.active_data_block = restore_active(
            source.active_data_block, BlockRole::Data);
        plane.active_mapping_block = restore_active(
            source.active_mapping_block, BlockRole::Mapping);
        plane.active_gc_block = restore_active(
            source.active_gc_block, BlockRole::GC);
    }

    if (image.compact_image) {
        const auto& source = *image.compact_image;
        const auto stacks = static_cast<std::uint64_t>(config_.device.stacks);
        if (source.page_count == 0 || source.page_count > total_pages_ ||
            source.data_blocks_by_plane.size() != planes_.size() ||
            source.mapping_ppns.size() != source.vpn_slot_count ||
            source.vpn_ranges.size() != source.vpn_slot_count ||
            source.mapping_page_count > source.vpn_slot_count ||
            source.retired_lpns.size() > source.page_count ||
            source.retired_mapping_vpns.size() >
                source.mapping_page_count ||
            (!source.mutable_image &&
             (!source.retired_lpns.empty() ||
              !source.retired_mapping_vpns.empty()))) {
            throw std::runtime_error(
                "HBF persistent compact image dimensions are inconsistent");
        }
        const auto last_lpn = checked_add(
            source.first_lpn,
            source.page_count - 1,
            "HBF restored compact last LPN");
        if (is_metadata_lpn(last_lpn)) {
            throw std::runtime_error(
                "HBF persistent compact image enters metadata LPN space");
        }
        const auto first_group =
            (source.first_lpn / stacks) /
            config_.host.mapping_entries_per_page;
        const auto last_group =
            (last_lpn / stacks) /
            config_.host.mapping_entries_per_page;
        const auto expected_first_vpn = checked_mul(
            first_group, stacks, "HBF restored compact first VPN");
        const auto expected_vpn_slots = checked_mul(
            checked_add(
                last_group - first_group,
                1,
                "HBF restored compact mapping groups"),
            stacks,
            "HBF restored compact VPN slots");
        if (source.first_vpn != expected_first_vpn ||
            source.vpn_slot_count != expected_vpn_slots) {
            throw std::runtime_error(
                "HBF persistent compact image VPN geometry diverged");
        }

        CompactLogicalImage compact;
        compact.mutable_image = source.mutable_image;
        compact.first_lpn = source.first_lpn;
        compact.page_count = source.page_count;
        compact.first_vpn = source.first_vpn;
        compact.vpn_slot_count = source.vpn_slot_count;
        compact.mapping_page_count = source.mapping_page_count;
        compact.data_blocks_by_plane = source.data_blocks_by_plane;
        compact.mapping_ppns = source.mapping_ppns;
        compact.vpn_ranges.reserve(source.vpn_ranges.size());
        for (const auto& range : source.vpn_ranges) {
            if (range.first_entry > config_.host.mapping_entries_per_page ||
                range.page_count > config_.host.mapping_entries_per_page ||
                range.first_entry + range.page_count >
                    config_.host.mapping_entries_per_page) {
                throw std::runtime_error(
                    "HBF persistent compact VPN range is out of bounds");
            }
            compact.vpn_ranges.push_back({
                .first_entry = range.first_entry,
                .page_count = range.page_count,
                .stack_page_offset = range.stack_page_offset,
            });
        }
        compact.vpn_offsets_by_stack.resize(config_.device.stacks);

        for (std::size_t plane = 0;
             plane < compact.data_blocks_by_plane.size();
             ++plane) {
            const auto& assigned = compact.data_blocks_by_plane[plane];
            if (assigned.size() > config_.device.blocks_per_plane) {
                throw std::runtime_error(
                    "HBF persistent compact data directory exceeds a plane");
            }
            for (std::size_t ordinal = 0;
                 ordinal < assigned.size();
                 ++ordinal) {
                const auto block = assigned[ordinal];
                if (block >= blocks_.size() ||
                    block_plane_index(static_cast<std::size_t>(block)) !=
                        plane ||
                    !compact.data_block_locations.emplace(
                        block,
                        CompactLogicalImage::DataBlockLocation{
                            .plane = plane,
                            .block_ordinal = ordinal,
                        }).second) {
                    throw std::runtime_error(
                        "HBF persistent compact data-block directory is invalid");
                }
            }
        }

        std::uint64_t active_mapping_pages = 0;
        std::uint64_t represented_data_pages = 0;
        for (std::size_t offset = 0;
             offset < compact.mapping_ppns.size();
             ++offset) {
            const auto& mapping_ppn = compact.mapping_ppns[offset];
            const auto& range = compact.vpn_ranges[offset];
            const bool needs_mapping = range.page_count != 0;
            if (needs_mapping != mapping_ppn.has_value()) {
                throw std::runtime_error(
                    "HBF persistent compact mapping slot activity diverged");
            }
            const auto mapping_vpn = checked_add(
                compact.first_vpn, offset, "HBF restored compact mapping VPN");
            if (range.page_count != 0) {
                compact.vpn_offsets_by_stack.at(
                    stack_for_vpn(mapping_vpn)).push_back(offset);
            }
            represented_data_pages = checked_add(
                represented_data_pages, range.page_count,
                "HBF restored compact represented data pages");
            if (!mapping_ppn) {
                continue;
            }
            if (*mapping_ppn >= total_pages_) {
                throw std::runtime_error(
                    "HBF persistent compact mapping PPN is out of range");
            }
            if (!compact.mapping_vpn_by_ppn.emplace(
                    *mapping_ppn, mapping_vpn).second) {
                throw std::runtime_error(
                    "HBF persistent compact mapping PPN is duplicated");
            }
            active_mapping_pages = checked_add(
                active_mapping_pages,
                1,
                "HBF restored compact mapping-page count");
        }
        if (active_mapping_pages != compact.mapping_page_count ||
            represented_data_pages != compact.page_count) {
            throw std::runtime_error(
                "HBF persistent compact mapping directory does not conserve pages");
        }

        const auto restore_live_blocks = [this](
            const std::vector<HbfPersistentCompactLiveBlock>& source_blocks,
            auto& destination,
            const char* description) {
            for (const auto& entry : source_blocks) {
                if (entry.block >= blocks_.size() || entry.live_pages == 0 ||
                    entry.live_pages > config_.device.pages_per_block ||
                    !destination.emplace(
                        entry.block, entry.live_pages).second) {
                    throw std::runtime_error(
                        std::string("HBF persistent compact ") + description +
                        " live-block index is invalid");
                }
            }
        };
        restore_live_blocks(
            source.live_data_pages_by_block,
            compact.live_data_pages_by_block,
            "data");
        restore_live_blocks(
            source.live_mapping_pages_by_block,
            compact.live_mapping_pages_by_block,
            "mapping");
        source.retired_lpns.for_each_ordered([&](std::uint64_t lpn) {
            if (lpn < compact.first_lpn ||
                lpn - compact.first_lpn >= compact.page_count ||
                !compact.retired_lpns.insert(lpn)) {
                throw std::runtime_error(
                    "HBF persistent compact retired LPN index is invalid");
            }
        });
        source.retired_mapping_vpns.for_each_ordered([&](std::uint64_t vpn) {
            if (vpn < compact.first_vpn ||
                vpn - compact.first_vpn >= compact.vpn_slot_count ||
                !compact.mapping_ppns[
                    static_cast<std::size_t>(vpn - compact.first_vpn)] ||
                !compact.retired_mapping_vpns.insert(vpn)) {
                throw std::runtime_error(
                    "HBF persistent compact retired VPN index is invalid");
            }
        });
        for (const auto& [lpn, _] : lpn_to_ppn_) {
            if (lpn >= compact.first_lpn &&
                lpn - compact.first_lpn < compact.page_count &&
                !compact.retired_lpns.contains(lpn)) {
                throw std::runtime_error(
                    "HBF persistent image duplicates a live compact LPN");
            }
        }
        for (const auto& [vpn, _] : mapping_vpn_to_ppn_) {
            if (vpn >= compact.first_vpn &&
                vpn - compact.first_vpn < compact.vpn_slot_count &&
                compact.mapping_ppns[
                    static_cast<std::size_t>(vpn - compact.first_vpn)] &&
                !compact.retired_mapping_vpns.contains(vpn)) {
                throw std::runtime_error(
                    "HBF persistent image duplicates a live compact VPN");
            }
        }
        const auto compact_ppn_without_validity = [this, &compact](
            std::uint64_t lpn) {
            const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
            if (mapping_vpn < compact.first_vpn ||
                mapping_vpn - compact.first_vpn >=
                    compact.vpn_slot_count) {
                throw std::runtime_error(
                    "HBF restored compact LPN is outside its VPN directory");
            }
            const auto vpn_offset = mapping_vpn - compact.first_vpn;
            const auto& range = compact.vpn_ranges.at(
                static_cast<std::size_t>(vpn_offset));
            const auto local_entry =
                (lpn / config_.device.stacks) %
                config_.host.mapping_entries_per_page;
            if (local_entry < range.first_entry ||
                local_entry - range.first_entry >= range.page_count) {
                throw std::runtime_error(
                    "HBF restored compact LPN is outside its VPN range");
            }
            const auto data_index = checked_add(
                range.stack_page_offset,
                local_entry - range.first_entry,
                "HBF restored compact data index");
            const auto pps = static_cast<std::uint64_t>(planes_per_stack());
            const auto stack = stack_for_vpn(mapping_vpn);
            const auto plane = stack * planes_per_stack() +
                static_cast<std::size_t>(data_index % pps);
            const auto page_ordinal = data_index / pps;
            const auto block_ordinal =
                page_ordinal / config_.device.pages_per_block;
            const auto page = static_cast<std::uint32_t>(
                page_ordinal % config_.device.pages_per_block);
            const auto& assigned = compact.data_blocks_by_plane.at(plane);
            if (block_ordinal >= assigned.size()) {
                throw std::runtime_error(
                    "HBF restored compact LPN exceeds its data directory");
            }
            return checked_add(
                checked_mul(
                    assigned[static_cast<std::size_t>(block_ordinal)],
                    config_.device.pages_per_block,
                    "HBF restored compact block PPN"),
                page,
                "HBF restored compact page PPN");
        };
        std::unordered_map<std::uint64_t, std::uint32_t>
            retired_data_pages_by_block;
        compact.retired_lpns.for_each([&](std::uint64_t lpn) {
            const auto ppn = compact_ppn_without_validity(lpn);
            auto& count = retired_data_pages_by_block[
                ppn / config_.device.pages_per_block];
            if (count == std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "HBF restored compact retired-page count overflowed");
            }
            ++count;
        });
        for (const auto& assigned : compact.data_blocks_by_plane) {
            for (const auto block_number : assigned) {
                const auto live = compact.live_data_pages_by_block.find(
                    block_number);
                const auto retired = retired_data_pages_by_block.find(
                    block_number);
                const auto original_pages = checked_add(
                    live == compact.live_data_pages_by_block.end() ?
                        0 : live->second,
                    retired == retired_data_pages_by_block.end() ?
                        0 : retired->second,
                    "HBF restored compact original block pages");
                if (original_pages > config_.device.pages_per_block) {
                    throw std::runtime_error(
                        "HBF restored compact block exceeds page geometry");
                }
                if (original_pages != 0) {
                    blocks_[static_cast<std::size_t>(block_number)]
                        .set_valid_range(
                            0, static_cast<std::uint32_t>(original_pages), config_.device.pages_per_block);
                }
            }
        }
        compact.retired_lpns.for_each([&](std::uint64_t lpn) {
            const auto ppn = compact_ppn_without_validity(lpn);
            blocks_[static_cast<std::size_t>(
                ppn / config_.device.pages_per_block)].clear_valid(
                    static_cast<std::uint32_t>(
                        ppn % config_.device.pages_per_block));
        });
        for (std::size_t offset = 0;
             offset < compact.mapping_ppns.size();
             ++offset) {
            if (!compact.mapping_ppns[offset]) {
                continue;
            }
            const auto vpn = checked_add(
                compact.first_vpn,
                offset,
                "HBF restored compact bitmap VPN");
            if (compact.retired_mapping_vpns.contains(vpn)) {
                continue;
            }
            const auto ppn = *compact.mapping_ppns[offset];
            blocks_[static_cast<std::size_t>(
                ppn / config_.device.pages_per_block)].set_valid(
                    static_cast<std::uint32_t>(
                        ppn % config_.device.pages_per_block), config_.device.pages_per_block);
        }
        compact_logical_image_ = std::move(compact);
        source.retired_lpns.for_each_ordered([&](std::uint64_t lpn) {
            record_mutated_lpn_range(lpn, 1);
        });
        // A compact-owned PPN can have been erased and reused by a later
        // materialized generation. Reapply current materialized validity
        // after reconstructing and retiring the original compact prefixes.
        for (const auto& [ppn, page] : programmed_pages_) {
            if (page.status == PageStatus::Valid ||
                page.status == PageStatus::StaticReadOnly) {
                blocks_[static_cast<std::size_t>(
                    ppn / config_.device.pages_per_block)].set_valid(
                        static_cast<std::uint32_t>(
                            ppn % config_.device.pages_per_block), config_.device.pages_per_block);
            }
        }
    }

    free_pages_ = image.state.free_pages;
    free_pages_per_stack_ = image.state.free_pages_per_stack;
    // Volatile search indexes are rebuilt once from the restored media,
    // without changing the persisted format or its physical state.
    for (auto& histogram : managed_wear_by_stack_) histogram.clear();
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        add_managed_block_wear(i);
        mark_gc_candidate_dirty(i);
    }
    restored_block_erases_ = 0;
    stats_.static_reserved_pages = 0;
    stats_.raw_reserved_pages = 0;
    for (const auto& block : blocks_) {
        restored_block_erases_ = checked_add(
            restored_block_erases_,
            block.erase_count,
            "HBF restored block erase count");
        if (block.role == BlockRole::StaticReadOnly) {
            stats_.static_reserved_pages = checked_add(
                stats_.static_reserved_pages,
                config_.device.pages_per_block,
                "HBF restored static page count");
        } else if (block.role == BlockRole::RawPhysical) {
            stats_.raw_reserved_pages = checked_add(
                stats_.raw_reserved_pages,
                config_.device.pages_per_block,
                "HBF restored raw reserved-page count");
        }
    }
    stats_.compact_initial_logical_data_pages =
        compact_logical_image_ ? compact_logical_image_->page_count : 0;
    stats_.compact_initial_mapping_pages =
        compact_logical_image_ ? compact_logical_image_->mapping_page_count : 0;
    stats_.initial_logical_data_pages = logical_mapping_entry_count();
    const auto compact_live_mapping_pages = compact_logical_image_ ?
        compact_logical_image_->mapping_page_count -
            compact_logical_image_->retired_mapping_vpns.size() :
        0;
    stats_.initial_mapping_pages = checked_add(
        mapping_vpn_to_ppn_.size(),
        compact_live_mapping_pages,
        "HBF restored initial mapping pages");
    stats_.free_pages = free_pages_;
    // Static/raw roles came from the image: derive or validate the logical
    // capacity now and check that every restored logical page lies inside it.
    logical_capacity_pages_ = config_.host.logical_capacity_bytes == 0 ?
        derive_logical_capacity_pages() :
        config_.host.logical_capacity_bytes / config_.device.page_size_bytes;
    resolve_logical_capacity();
    for (const auto& [lpn, _] : lpn_to_ppn_) {
        if (lpn >= logical_capacity_pages_) {
            throw std::runtime_error(
                "HBF persistent image maps a logical page beyond the logical "
                "capacity of " + std::to_string(logical_capacity_pages_) + " pages");
        }
    }
    if (compact_logical_image_ && compact_logical_image_->page_count != 0 &&
        compact_logical_image_->first_lpn + compact_logical_image_->page_count - 1 >=
            logical_capacity_pages_) {
        throw std::runtime_error(
            "HBF persistent compact image exceeds the logical capacity of " +
            std::to_string(logical_capacity_pages_) + " pages");
    }
    // WL controller state is volatile. An image restart cannot manufacture
    // budget credit or forget a possible recent migration: conservatively
    // hold every restored block for one full cooldown interval.
    if (config_.host.static_wear_leveling_cooldown_erases != 0) {
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            wear_leveling_last_move_by_block_[i] =
                wear_leveling_erase_clock(stack_of_block(i));
        }
    }
    // Thermal state is runtime-only: a restored image boots a fresh process
    // at the configured boot state with an empty pacing calendar.
    for (auto& stack : thermal_stacks_) {
        stack.node = ThermalNodeState{};
        stack.node.temperature_c = thermal_boot_temperature_c_;
        stack.node.peak_c = thermal_boot_temperature_c_;
        stack.node.throttled = config_.device.thermal_start_at_ceiling;
        stack.pacing = ResourceTimeline{};
    }
    refresh_parallel_stats();
    media_->restore_zone_mapping(image.zone_remapping);
    zone_managed_ = image.zone_managed;
    // A zone-only image has no frozen logical namespace. Keep unassigned
    // zones available to its host after restart, just as before checkpoint.
    if (zone_managed_ && lpn_to_ppn_.empty() && mapping_vpn_to_ppn_.empty() && !compact_logical_image_)
        logical_capacity_frozen_ = false;
    initial_erase_counts_.clear();
    initial_erase_counts_.reserve(blocks_.size());
    for (const auto& block : blocks_) initial_erase_counts_.push_back(block.erase_count);
    if (mapping_policy_) mapping_policy_->restore(*this, image.host_mapping_state);
}

void HbfController::seed_media_image() {
    if (media_image_seeded_) return;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        media_->seed_block(b, blocks_[b].next_page, blocks_[b].erase_count);
        // Before the first command every unprogrammed block (free, or held
        // by a raw zone) is still in its installed state.
        if (config_.host.initial_free_blocks_erased && blocks_[b].next_page == 0 &&
            blocks_[b].role != BlockRole::StaticReadOnly)
            media_->seed_erased_block(b);
    }
    stats_.preconditioned_erased_blocks = media_->preconditioned_erased_blocks();
    media_image_seeded_ = true;
}

PhysicalCompletion HbfController::issue(const PhysicalRequest& request) {
    if (mapping_policy_) {
        if (request.address_space != AddressSpace::Logical)
            throw std::invalid_argument("host mapping owns the media; use logical addresses");
        return mapping_policy_->issue(*this, request);
    }
    return issue_media(request);
}

PhysicalCompletion HbfController::issue_media(const PhysicalRequest& request, double data_ready_ns) {
    if (mapping_is_local() && request.address_space != AddressSpace::Logical)
        throw std::runtime_error("device-local mapping control accepts logical requests only");
    seed_media_image();
    if (config_.host.mapping_mode == MappingMode::RawPhysical &&
        request.address_space == AddressSpace::Logical) {
        throw std::runtime_error("raw-physical media requires the OCP channel frontend or explicit physical addresses");
    }
    if (request.tier != Tier::HBF) {
        throw std::runtime_error("HbfController received non-HBF request");
    }
    // Validate provenance before any request state changes. A malformed enum
    // must not be retained in a deferred buffer and fail only at a later
    // eviction or drain.
    (void)resolve_heatmap_source(
        TransactionSource::User, request.heatmap_source);
    if (request.bytes == 0 && request.op != Op::Erase) {
        throw std::runtime_error("HBF request bytes must be positive");
    }
    if (request.op != Op::Erase &&
        request.bytes - 1 > std::numeric_limits<std::uint64_t>::max() - request.addr) {
        throw std::runtime_error("HBF request address range overflows uint64_t");
    }
    if (request.op == Op::Refresh) {
        throw std::runtime_error("HBF v0 does not model background refresh");
    }
    if (!std::isfinite(request.arrival_ns) || request.arrival_ns < 0.0) {
        throw std::runtime_error("HBF request arrival must be finite and non-negative");
    }
    const bool static_request = request.address_space == AddressSpace::Static;
    const bool physical_request = request.address_space == AddressSpace::Physical ||
        request.op == Op::Erase;
    const bool direct_media_request = physical_request || static_request;
    if (compact_logical_image_ &&
        !compact_logical_image_->mutable_image &&
        (request.op != Op::Read || direct_media_request)) {
        throw std::runtime_error(
            "HBF compact read-only initial image accepts logical reads only");
    }
    if (!direct_media_request) {
        const auto last_addr = request.addr + (request.bytes - 1);
        require_logical_capacity(last_addr / config_.device.page_size_bytes);
    } else if (physical_request) {
        const auto capacity_bytes = checked_mul(total_pages_, config_.device.page_size_bytes,
            "HBF physical capacity bytes");
        if (request.addr >= capacity_bytes ||
            (request.op != Op::Erase && request.bytes > capacity_bytes - request.addr)) {
            throw std::runtime_error("HBF physical request range is out of capacity");
        }
    } else {
        const auto capacity_bytes = checked_mul(
            total_pages_, config_.device.page_size_bytes,
            "HBF static source capacity bytes");
        if (request.op != Op::Read || request.addr >= capacity_bytes ||
            request.bytes > capacity_bytes - request.addr) {
            throw std::runtime_error(
                "HBF static request must be an in-capacity read");
        }
        const auto first_source_page =
            request.addr / config_.device.page_size_bytes;
        const auto source_pages = page_count_for(
            logical_page_address(request.addr), request.bytes);
        const auto spans = static_page_run_spans(
            first_source_page, source_pages);
        for (const auto& span : spans) {
            const auto plane_base = span.plane * config_.device.blocks_per_plane;
            const auto first_block = span.first_page /
                config_.device.pages_per_block;
            const auto last_block =
                (span.first_page + span.page_count - 1) /
                config_.device.pages_per_block;
            for (auto block = first_block; block <= last_block; ++block) {
                if (blocks_.at(plane_base + block).role !=
                    BlockRole::StaticReadOnly) {
                    throw std::runtime_error(
                        "HBF static request escapes its reserved extent");
                }
            }
        }
    }
    if (last_issue_arrival_ns_ && request.arrival_ns < *last_issue_arrival_ns_) {
        throw std::runtime_error(
            "HBF requests and drains must be issued in nondecreasing arrival order; "
            "enqueue/sort the workload before simulation so state cannot travel backward "
            "in time");
    }
    // Once a request reaches state-dependent validation, its arrival becomes
    // a causal barrier even if the target is rejected. This prevents a caught
    // future-time error from being followed by an earlier request after prior
    // commits have already been materialized.
    last_issue_arrival_ns_ = request.arrival_ns;
    reservation_causal_watermark_ns_ = request.arrival_ns;
    media_->advance_cache(request.arrival_ns);
    prune_expired_state(request.arrival_ns);
    const auto request_stack = physical_request ?
        decode(request.addr).stack :
        static_cast<std::uint32_t>(
            stack_for_lpn(request.addr / config_.device.page_size_bytes));
    std::vector<bool> touched_stacks(config_.device.stacks, false);
    touched_stacks.at(request_stack) = true;
    if (request.op != Op::Erase && physical_request) {
        const auto last_stack = decode(request.addr + request.bytes - 1).stack;
        for (std::size_t stack = request_stack; stack <= last_stack; ++stack) {
            touched_stacks.at(stack) = true;
        }
    } else if (static_request) {
        const auto first_source_page =
            request.addr / config_.device.page_size_bytes;
        const auto source_pages = page_count_for(
            logical_page_address(request.addr), request.bytes);
        if (source_pages >= 2 * static_cast<std::uint64_t>(config_.device.stacks)) {
            std::fill(touched_stacks.begin(), touched_stacks.end(), true);
        } else {
            for (std::uint64_t index = 0; index < source_pages; ++index) {
                touched_stacks.at(
                    stack_for_lpn(first_source_page + index)) = true;
            }
        }
    } else if (!physical_request && config_.device.stacks > 1) {
        const auto first_lpn = request.addr / config_.device.page_size_bytes;
        const auto last_lpn =
            (request.addr + request.bytes - 1) / config_.device.page_size_bytes;
        std::size_t remaining_stacks = config_.device.stacks - 1;
        for (auto lpn = first_lpn;; ++lpn) {
            const auto stack = stack_for_lpn(lpn);
            if (!touched_stacks.at(stack)) {
                touched_stacks[stack] = true;
                --remaining_stacks;
            }
            if (remaining_stacks == 0 || lpn == last_lpn) {
                break;
            }
        }
    }
    std::vector<double> state_ready_by_stack(
        config_.device.stacks, request.arrival_ns);
    std::vector<std::unordered_set<std::uint64_t>> selected_commits_by_stack(
        config_.device.stacks);
    if (physical_request) {
        const auto first_block = block_index(decode(request.addr));
        const auto last_block = request.op == Op::Erase ? first_block :
            block_index(decode(request.addr + request.bytes - 1));
        for (auto block = first_block; block <= last_block; ++block) {
            const auto stack = stack_of_block(block);
            state_ready_by_stack[stack] = std::max(
                state_ready_by_stack[stack],
                materialized_ready_by_block_.at(block));
            const auto pending = pending_block_transitions_.find(block);
            if (pending != pending_block_transitions_.end()) {
                state_ready_by_stack[stack] = std::max(
                    state_ready_by_stack[stack], pending->second.finish_ns);
                add_pending_block_transition_commits(
                    block, selected_commits_by_stack[stack]);
            }
            if (request.op == Op::Erase) {
                // A GC source reclaim depends on every relocation destination.
                // A later raw erase of one of those destination blocks must
                // therefore materialize the source chain before it retires
                // the destination epoch; otherwise the GC publication would
                // be suppressed and the source erase would lose live data.
                for (const auto& [source_block, source_erase] :
                     pending_block_transitions_) {
                    if (!source_erase.garbage_collection ||
                        !std::any_of(
                            source_erase.destination_ppns.begin(),
                            source_erase.destination_ppns.end(),
                            [this, block](std::uint64_t destination_ppn) {
                                return destination_ppn /
                                        config_.device.pages_per_block == block;
                            })) {
                        continue;
                    }
                    state_ready_by_stack[stack] = std::max(
                        state_ready_by_stack[stack],
                        source_erase.finish_ns);
                    add_pending_block_transition_commits(
                        source_block,
                        selected_commits_by_stack[stack]);
                }
            }
        }
        if (request.op != Op::Erase) {
            const auto first_ppn = encode_ppn(decode(request.addr));
            const auto pages = page_count_for(decode(request.addr), request.bytes);
            for (std::uint64_t i = 0; i < pages; ++i) {
                const auto ppn = first_ppn + i;
                const auto stack = stack_of_block(static_cast<std::size_t>(
                    ppn / config_.device.pages_per_block));
                if (const auto observed = materialized_ready_by_ppn_.find(ppn);
                    observed != materialized_ready_by_ppn_.end()) {
                    state_ready_by_stack[stack] = std::max(
                        state_ready_by_stack[stack], observed->second);
                }
                if (const auto program = pending_physical_programs_.find(ppn);
                    program != pending_physical_programs_.end()) {
                    state_ready_by_stack[stack] = std::max(
                        state_ready_by_stack[stack], program->second.commit_ns);
                    selected_commits_by_stack[stack].insert(
                        program->second.commit_sequence);
                }
            }
        }
    } else if (!static_request &&
               (!materialized_ready_by_lpn_.empty() ||
                !materialized_ready_by_vpn_.empty())) {
        const auto first_lpn = request.addr / config_.device.page_size_bytes;
        const auto last_lpn =
            (request.addr + request.bytes - 1) / config_.device.page_size_bytes;
        for (auto lpn = first_lpn; lpn <= last_lpn; ++lpn) {
            const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
            const auto stack = stack_for_vpn(mapping_vpn);
            if (const auto observed = materialized_ready_by_lpn_.find(lpn);
                observed != materialized_ready_by_lpn_.end()) {
                state_ready_by_stack[stack] = std::max(
                    state_ready_by_stack[stack], observed->second);
            }
            if (const auto observed = materialized_ready_by_vpn_.find(mapping_vpn);
                observed != materialized_ready_by_vpn_.end()) {
                state_ready_by_stack[stack] = std::max(
                    state_ready_by_stack[stack], observed->second);
            }
        }
    }
    // Materialize every causally prior state transition before validating a
    // physical target. Otherwise a completed-but-not-yet-applied program
    // could make the same PPN appear erased and permit illegal reprogramming.
    // Globally materialize only events that have truly completed by host
    // arrival (or a whole-stack causal GC/drain barrier). Future block/LPN/VPN
    // dependencies apply only their selected callback chain; applying the
    // whole stack would make independent planes depend on API call order.
    apply_commits_through(request.arrival_ns);
    for (std::size_t stack = 0; stack < config_.device.stacks; ++stack) {
        if (!touched_stacks[stack]) {
            continue;
        }
        const double causal_ready_ns = causal_state_ready_by_stack_.at(stack);
        if (causal_ready_ns > request.arrival_ns) {
            apply_stack_commits_through(stack, causal_ready_ns);
        }
        const double ready_ns = std::max(
            state_ready_by_stack[stack], causal_ready_ns);
        state_ready_by_stack[stack] = ready_ns;
        apply_selected_commits_through(
            selected_commits_by_stack[stack], ready_ns);
        state_observation_by_stack_.at(stack) = ready_ns;
    }
    const double ingress_state_ready_ns = state_ready_by_stack.at(request_stack);
    if (physical_request && request.op == Op::Write) {
        const auto first = encode_ppn(decode(request.addr));
        const auto pages = page_count_for(decode(request.addr), request.bytes);
        reserve_physical_program_range(first, pages);
    } else if (physical_request && request.op == Op::Erase) {
        const auto block = block_index(decode(request.addr));
        if (pending_block_transitions_.contains(block) ||
            blocks_.at(block).erase_pending) {
            throw std::runtime_error("HBF block already has an in-flight physical erase");
        }
        if (blocks_.at(block).role == BlockRole::StaticReadOnly) {
            throw std::runtime_error("HBF erase targets a static read-only block");
        }
    }
    stats_.first_arrival_ns = std::min(stats_.first_arrival_ns, request.arrival_ns);

    PhysicalCompletion out;
    out.id = request.id;
    out.tier = Tier::HBF;
    out.op = request.op;
    out.arrival_ns = request.arrival_ns;
    out.logical_bytes = request.bytes;
    auto* trace_spans = trace_spans_enabled(request.trace) ? &out.spans : nullptr;
    const auto physical_bytes_before = stats_.physical_read_bytes + stats_.physical_write_bytes;
    if (ingress_state_ready_ns > request.arrival_ns) {
        out.breakdown.scheduler_queue_wait_ns +=
            ingress_state_ready_ns - request.arrival_ns;
        if (trace_spans != nullptr) {
            trace_wait(
                trace_spans,
                logic_entity(static_cast<std::uint32_t>(request_stack)),
                request.arrival_ns,
                ingress_state_ready_ns,
                "wait_stack_state_dependency");
        }
    }

    const auto ingress_stack = request_stack;
    auto& ingress_logic = logic_dies_.at(ingress_stack);
    const double request_command_done = ingress_state_ready_ns;
    const auto ingress_slot = reserve(
        request_command_done,
        config_.device.logic_scheduler_issue_ns,
        ingress_logic.ingress);
    const double logic_start = ingress_slot.start_ns;
    out.breakdown.ingress_queue_wait_ns = ingress_slot.wait_ns;
    double issued_ns = ingress_slot.finish_ns;
    out.breakdown.command_ns += config_.device.logic_scheduler_issue_ns;
    if (trace_spans != nullptr) {
        const auto entity = logic_entity(ingress_stack);
        add_trace_span(
            trace_spans,
            "logic_die_queue",
            "queue",
            entity,
            request_command_done,
            logic_start);
        add_trace_span(
            trace_spans,
            "logic_scheduler_issue",
            "logic",
            entity,
            logic_start,
            issued_ns);
    }
    out.start_ns = logic_start;
    const auto wait_for_page_stack = [&](std::size_t stack, double earliest_ns) {
        const double ready_ns = std::max(
            earliest_ns, state_ready_by_stack.at(stack));
        if (ready_ns > earliest_ns) {
            out.breakdown.scheduler_queue_wait_ns += ready_ns - earliest_ns;
            if (trace_spans != nullptr) {
                trace_wait(
                    trace_spans,
                    logic_entity(static_cast<std::uint32_t>(stack)),
                    earliest_ns,
                    ready_ns,
                    "wait_page_stack_state_dependency");
            }
        }
        return ready_ns;
    };

    if (request.op == Op::Read) {
        const auto first_lpn = physical_request ? 0 : request.addr / config_.device.page_size_bytes;
        const auto logical = physical_request ? decode(request.addr) : logical_page_address(request.addr);
        const auto pages = physical_request ? page_count_for(decode(request.addr), request.bytes) :
            page_count_for(logical, request.bytes);

        double finish_ns = issued_ns;
        std::uint64_t media_pages = 0;
        std::uint64_t read_buffer_pages = 0;
        std::uint64_t write_buffer_pages = 0;
        std::uint64_t erased_pages = 0;
        std::string first_path;
        if (pages > 1) {
            const double split_ready_ns = issued_ns;
            const auto split = reserve(
                split_ready_ns,
                config_.device.address_generation_ns,
                ingress_logic.ingress);
            out.breakdown.ingress_queue_wait_ns += split.wait_ns;
            if (trace_spans != nullptr) {
                const auto entity = logic_entity(ingress_stack);
                trace_wait(
                    trace_spans,
                    entity,
                    split_ready_ns,
                    split.start_ns,
                    "wait_read_split_ingress");
                add_trace_span(
                    trace_spans,
                    "read_split",
                    "logic",
                    entity,
                    split.start_ns,
                    split.finish_ns,
                    true,
                    std::to_string(pages) + " page transactions");
            }
            out.breakdown.address_mapping_ns += config_.device.address_generation_ns;
            // Address generation is real ingress work. Downstream page
            // transactions cannot observe the split before its actual
            // reservation (including any queueing) has completed.
            issued_ns = split.finish_ns;
            stats_.read_splits++;
            stats_.read_split_pages += pages;
        }
        const auto first_ppn = physical_request ?
            encode_ppn(decode(request.addr)) : 0;
        ++stats_.scalar_read_requests;
        stats_.scalar_read_pages += pages;
        // Request-granular thermal pacing: one pacing decision per touched
        // stack per scalar request, covering all requested pages. Pages
        // served from buffers consume pacing budget conservatively here
        // (timing can only err slower) while heat deposits stay per actual
        // media page, so thermal_media_energy_j remains exact media work.
        std::vector<double> thermal_scalar_ready_ns;
        double thermal_budget_finish_ns = issued_ns;
        if (config_.device.thermal_enabled && pages != 0) {
            thermal_scalar_ready_ns.assign(config_.device.stacks, issued_ns);
            std::vector<std::uint64_t> thermal_pages_by_stack(
                config_.device.stacks, 0);
            if (config_.device.stacks == 1) {
                thermal_pages_by_stack[0] = pages;
            } else {
                for (std::uint64_t i = 0; i < pages; ++i) {
                    const auto stack = physical_request ?
                        stack_of_block(static_cast<std::size_t>(
                            (first_ppn + i) / config_.device.pages_per_block)) :
                        stack_for_lpn(first_lpn + i);
                    thermal_pages_by_stack.at(stack)++;
                }
            }
            for (std::size_t stack = 0;
                 stack < thermal_pages_by_stack.size();
                 ++stack) {
                const auto count = thermal_pages_by_stack[stack];
                if (count == 0) {
                    continue;
                }
                const auto admission = thermal_pace_media(
                    stack,
                    issued_ns,
                    static_cast<double>(count) * thermal_read_energy_j_,
                    count,
                    out.breakdown,
                    trace_spans);
                thermal_scalar_ready_ns[stack] = admission.ready_ns;
                thermal_budget_finish_ns = std::max(
                    thermal_budget_finish_ns, admission.budget_finish_ns);
            }
        }
        // Traced/metadata paths retain the general scheduler: they need event
        // spans or non-foreground payload routes. Request thermal pacing above
        // remains intact and the group kernel deposits each page's energy.
        // This is execution batching only; arrivals, credits and cache decisions
        // stay in page order, including between consecutive bank-sized groups.
        const bool grouped_pipeline = pages > 1 && trace_spans == nullptr &&
            address_heatmap_ == nullptr && !mapping_is_local();
        const auto group_width = static_cast<std::uint64_t>(planes_.size());
        auto read_group = make_read_pipeline_group();
        CompactReadRun compact_run;
        const bool compact_run_eligible = grouped_pipeline &&
            !physical_request && !static_request && compact_logical_image_.has_value();
        for (std::uint64_t i = 0; i < pages; ++i) {
            if (i % group_width == 0) read_group.pages = 0;
            double page_ready_ns = issued_ns;
            const auto range_begin = i == 0 ? logical.offset : 0;
            const auto preceding_bytes = i == 0 ? std::uint64_t{0} :
                (config_.device.page_size_bytes - logical.offset) +
                    (i - 1) * config_.device.page_size_bytes;
            const auto range_bytes = std::min(
                request.bytes - preceding_bytes,
                config_.device.page_size_bytes - range_begin);
            const auto range_end = range_begin + range_bytes;
            const auto device_bytes = ((range_end + 63) / 64 - range_begin / 64) * 64;
            const auto lpn = first_lpn + i;
            const auto compact_address = compact_run_eligible ?
                compact_read_address(lpn, first_lpn + pages, compact_run) : std::nullopt;
            const auto* read_address = compact_address ? &*compact_address : nullptr;
            const auto page_stack = physical_request ?
                stack_of_block(static_cast<std::size_t>(
                    (encode_ppn(decode(request.addr)) + i) /
                    config_.device.pages_per_block)) :
                (read_address != nullptr ? read_address->stack : stack_for_lpn(lpn));
            if (!thermal_scalar_ready_ns.empty()) {
                page_ready_ns = std::max(
                    page_ready_ns, thermal_scalar_ready_ns[page_stack]);
            }
            page_ready_ns = wait_for_page_stack(page_stack, page_ready_ns);
            page_ready_ns = admit_foreground_page_read(
                page_stack,
                page_ready_ns,
                out.breakdown,
                trace_spans);
            const auto complete_page = [&](double page_finish_ns) {
                complete_foreground_page_read(page_stack, page_finish_ns);
                finish_ns = std::max(finish_ns, page_finish_ns);
            };
            std::optional<std::uint64_t> ppn;
            if (physical_request) {
                ppn = encode_ppn(decode(request.addr)) + i;
            } else if (static_request) {
                ppn = static_ppn_for_source_page(lpn);
            } else {
                auto& page_write_buffer = write_buffer(stack_for_lpn(lpn));
                const auto buffered = page_write_buffer.find(lpn);
                const std::vector<DirtyRange>* buffered_ranges = nullptr;
                double buffered_ready_ns = page_ready_ns;
                if (buffered != page_write_buffer.end()) {
                    buffered_ranges = &buffered->second.ranges;
                    buffered_ready_ns = buffered->second.ready_ns;
                } else if (const auto inflight = inflight_buffered_writes_.find(lpn);
                           inflight != inflight_buffered_writes_.end()) {
                    const InflightBufferedWrite* newest = nullptr;
                    for (const auto& generation : inflight->second) {
                        const auto target_block = static_cast<std::size_t>(
                            generation.target_ppn / config_.device.pages_per_block);
                        const auto& block = blocks_.at(target_block);
                        if (block.epoch == generation.target_block_epoch &&
                            !block.erase_pending &&
                            page_ready_ns < generation.commit_ns &&
                            (newest == nullptr || generation.generation > newest->generation)) {
                            newest = &generation;
                        }
                    }
                    if (newest != nullptr) {
                        buffered_ranges = &newest->ranges;
                        buffered_ready_ns = newest->ready_ns;
                    }
                }
                std::uint64_t buffered_overlap_bytes = 0;
                if (buffered_ranges != nullptr) {
                    for (const auto& range : *buffered_ranges) {
                        const auto lo = std::max(range.begin, range_begin);
                        const auto hi = std::min(range.end, range_end);
                        if (hi > lo) {
                            buffered_overlap_bytes += hi - lo;
                        }
                    }
                    // A dirty range elsewhere in the page is irrelevant to
                    // this sub-page read: it must not create a false WB hit,
                    // wait, overlay, or report classification.
                    if (buffered_overlap_bytes == 0) {
                        buffered_ranges = nullptr;
                    } else if (buffered_ready_ns > page_ready_ns) {
                        if (trace_spans != nullptr) {
                            trace_wait(
                                trace_spans,
                                logic_entity(static_cast<std::uint32_t>(
                                    stack_for_lpn(lpn))),
                                page_ready_ns,
                                buffered_ready_ns,
                                "wait_prior_write_buffer_stage");
                        }
                        out.breakdown.scheduler_queue_wait_ns +=
                            buffered_ready_ns - page_ready_ns;
                        page_ready_ns = buffered_ready_ns;
                    }
                }
                if (buffered_ranges != nullptr &&
                    dirty_ranges_cover(*buffered_ranges, range_begin, range_end)) {
                    complete_page(serve_read_from_write_buffer(
                        lpn,
                        range_begin,
                        range_end,
                        *buffered_ranges,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans));
                    if (first_path.empty()) {
                        first_path = "write_buffer/lpn" + std::to_string(lpn);
                    }
                    write_buffer_pages++;
                    continue;
                }
                ppn = lookup_lpn(
                    lpn,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    LookupIntent::ReadData,
                    read_address);
                if (buffered_ranges != nullptr && ppn && write_buffer_on_device()) {
                    // Device-side overlay: the old image (device-DRAM hit or
                    // NAND fill) and the dirty bytes merge in device DRAM,
                    // and only the requested bytes cross HBIO.
                    if (first_path.empty()) {
                        first_path = decode_ppn(*ppn).path();
                    }
                    const auto stack = stack_for_lpn(lpn);
                    if (!mapping_is_local())
                        page_ready_ns = schedule_external_request_command(
                            channel_index(decode_ppn(*ppn)), page_ready_ns, out.breakdown,
                            trace_spans, "device-DRAM overlay read", "user");
                    if (const auto cached = device_dram_probe(*ppn)) {
                        page_ready_ns = device_dram_transfer(stack, config_.device.page_size_bytes,
                            false, std::max(page_ready_ns, *cached), out.breakdown, trace_spans,
                            "device_dram_read_hit");
                        // The write buffer supplies its dirty bytes; the
                        // cached image serves the rest of the range.
                        stats_.device_dram_read_hit_bytes += range_bytes - buffered_overlap_bytes;
                        read_buffer_pages++;
                    } else {
                        // The request pays to bring the old image into DRAM
                        // (it is merged there), so the page is installed as
                        // a clean line without a background fill.
                        page_ready_ns = device_dram_transfer(stack, config_.device.page_size_bytes,
                            true, schedule_read_page(*ppn, page_ready_ns, out.breakdown, trace_spans,
                            TransactionSource::User, request.heatmap_source,
                            ReadPayloadRoute::DeviceDram, 0, nullptr, true),
                            out.breakdown, trace_spans, "device_dram_overlay_fill");
                        device_dram_install(*ppn, page_ready_ns);
                        media_pages++;
                    }
                    const auto overlay_detail = trace_spans == nullptr ?
                        std::string{} : "lpn" + std::to_string(lpn);
                    page_ready_ns = schedule_write_buffer_dram_access(stack,
                        buffered_overlap_bytes, false, page_ready_ns, out.breakdown, trace_spans,
                        "write_buffer_overlay_dram", overlay_detail);
                    complete_page(schedule_external_read_egress(channel_index(decode_ppn(*ppn)),
                        range_bytes, page_ready_ns, out.breakdown, trace_spans,
                        "write_buffer_overlay_hbio_out", "device-DRAM overlay"));
                    stats_.write_buffer_read_hits++;
                    stats_.write_buffer_read_bytes += buffered_overlap_bytes;
                    write_buffer_pages++;
                    continue;
                } else if (buffered_ranges != nullptr && ppn) {
                    if (first_path.empty()) {
                        first_path = decode_ppn(*ppn).path();
                    }
                    if (read_buffer_contains(*ppn, page_ready_ns)) {
                        page_ready_ns = serve_read_from_read_buffer(
                            *ppn,
                            range_bytes,
                            page_ready_ns,
                            out.breakdown,
                            trace_spans, true);
                        read_buffer_pages++;
                    } else {
                        stats_.read_buffer_misses++;
                        page_ready_ns = schedule_read_page(
                            *ppn,
                            page_ready_ns,
                            out.breakdown,
                            trace_spans,
                            TransactionSource::User,
                            request.heatmap_source,
                            ReadPayloadRoute::HostBuffer,
                            0,
                            nullptr,
                            true);
                        read_buffer_insert(*ppn, page_ready_ns);
                        media_pages++;
                    }
                    // The dirty bytes live in controller DRAM; reading them
                    // for the overlay uses the same pipelined DRAM resource
                    // the full-coverage path charges.
                    const auto overlay_detail = trace_spans == nullptr ?
                        std::string{} : "lpn" + std::to_string(lpn);
                    page_ready_ns = schedule_write_buffer_dram_access(
                        static_cast<std::uint32_t>(stack_for_lpn(lpn)),
                        buffered_overlap_bytes,
                        false,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans,
                        "write_buffer_overlay_dram",
                        overlay_detail);
                    page_ready_ns = host_memory_transfer(stack_for_lpn(lpn), buffered_overlap_bytes,
                        Op::Write, page_ready_ns, out.breakdown, trace_spans);
                    page_ready_ns = host_memory_transfer(stack_for_lpn(lpn), range_bytes,
                        Op::Read, page_ready_ns, out.breakdown, trace_spans);
                    complete_page(page_ready_ns);
                    stats_.write_buffer_read_hits++;
                    stats_.write_buffer_read_bytes += buffered_overlap_bytes;
                    write_buffer_pages++;
                    continue;
                } else if (buffered_ranges != nullptr) {
                    complete_page(serve_read_from_write_buffer(
                        lpn,
                        range_begin,
                        range_end,
                        *buffered_ranges,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans));
                    if (first_path.empty()) {
                        first_path = "write_buffer+erased/lpn" +
                            std::to_string(lpn);
                    }
                    write_buffer_pages++;
                    continue;
                }
            }
            if (!ppn) {
                // Host policy supplies the erased value for an unmapped page.
                // There is no device read or additional payload-memory port.
                complete_page(page_ready_ns);
                if (first_path.empty()) {
                    first_path = "erased/lpn" + std::to_string(lpn);
                }
                erased_pages++;
                continue;
            }
            if (first_path.empty()) {
                first_path = decode_ppn(*ppn).path();
            }
            // The device serves the closest copy: the NAND bank's decoded
            // page buffer, then the device DRAM read cache, then the array.
            // Physical-address probes bypass the device DRAM; FTL-managed
            // logical and static pages consult it.
            const bool device_cached = device_dram_enabled() && !physical_request;
            if (read_buffer_contains(*ppn, page_ready_ns)) {
                const auto served = serve_read_from_read_buffer(
                    *ppn,
                    device_bytes,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans);
                complete_page(served);
                if (device_cached) device_dram_fill(*ppn, served, trace_spans);
                read_buffer_pages++;
                continue;
            }
            if (device_cached) {
                if (const auto cached = device_dram_probe(*ppn)) {
                    complete_page(device_dram_serve_hit(*ppn, device_bytes, page_ready_ns,
                        *cached, out.breakdown, trace_spans));
                    read_buffer_pages++;
                    continue;
                }
            }
            stats_.read_buffer_misses++;
            double decoded_ready_ns = 0.0;
            const auto bank = static_cast<std::size_t>(*ppn /
                config_.device.pages_per_block / config_.device.blocks_per_plane);
            const auto& full_plane_windows = planes_[bank].full_plane_windows;
            // Completed program/erase windows are retained until an amortized
            // cleanup. Their presence must not permanently disable grouping.
            // Windows are ordered and disjoint; if the last has ended before
            // this page is ready, no stage of its read can intersect one.
            const bool group_page = grouped_pipeline &&
                (full_plane_windows.empty() ||
                 full_plane_windows.back().end_ns <= page_ready_ns);
            if (!group_page) read_group.pages = 0;
            const double page_finish = group_page ? schedule_read_group_page(
                *ppn, page_ready_ns, device_bytes, out.breakdown, decoded_ready_ns, read_group) :
                schedule_read_page(*ppn, page_ready_ns, out.breakdown, trace_spans,
                    TransactionSource::User, request.heatmap_source, ReadPayloadRoute::External,
                    device_bytes, &decoded_ready_ns, true);
            complete_page(page_finish);
            // The decoded page becomes cacheable when SRAM fill completes,
            // independently of how long its user's HBIO egress waits.
            read_buffer_insert(*ppn, decoded_ready_ns);
            if (device_cached) device_dram_fill(*ppn, decoded_ready_ns, trace_spans);
            media_pages++;
        }
        // A paced request may not complete before its energy has been
        // dissipated at the pacing budget for the complete admitted request.
        finish_ns = std::max(finish_ns, thermal_budget_finish_ns);
        out.finish_ns = finish_ns;
        if (write_buffer_pages > 0 && (media_pages > 0 || read_buffer_pages > 0)) {
            out.note = "mapped-read-with-write-buffer-overlay";
        } else if (write_buffer_pages > 0) {
            out.note = "write-buffer-read";
        } else if (erased_pages > 0 && media_pages == 0 && read_buffer_pages == 0) {
            out.note = pages == 1 ? "unmapped-erased-read" : "unmapped-erased-multi-page-read";
        } else if (erased_pages > 0) {
            out.note = "mixed-mapped-and-erased-read";
        } else if (read_buffer_pages > 0 && media_pages == 0) {
            out.note = pages == 1 ? "read-buffer-hit" : "read-buffer-multi-page-hit";
        } else {
            out.note = media_pages == 1 ? "mapped-page-read" : "mapped-multi-page-read";
        }
        if (physical_request) {
            out.resource_path = decode(request.addr).path();
        } else if (static_request) {
            out.resource_path = "static-page" + std::to_string(first_lpn) +
                "->" + decode_ppn(
                    static_ppn_for_source_page(first_lpn)).path();
        } else if (!first_path.empty()) {
            out.resource_path = "lpn" + std::to_string(first_lpn) + "->" + first_path;
        } else {
            out.resource_path = "lpn" + std::to_string(first_lpn) + "->unmapped";
        }

        stats_.read_requests++;
        if (request.address_space == AddressSpace::Logical) {
            stats_.logical_read_bytes += request.bytes;
        }
        stats_.physical_read_bytes += media_pages * config_.device.page_size_bytes;
        stats_.page_reads += media_pages;
    } else if (request.op == Op::Write) {
        const auto first_lpn = physical_request ? 0 : request.addr / config_.device.page_size_bytes;
        const auto logical = physical_request ? decode(request.addr) : logical_page_address(request.addr);
        const auto pages = physical_request ? page_count_for(decode(request.addr), request.bytes) :
            page_count_for(logical, request.bytes);

        if (!physical_request) {
            record_mutated_lpn_range(first_lpn, pages);
        }

        double finish_ns = issued_ns;
        std::uint64_t first_ppn = 0;
        std::string first_write_path;
        std::uint64_t remaining_bytes = request.bytes;
        for (std::uint64_t i = 0; i < pages; ++i) {
            double page_ready_ns = std::max(issued_ns, data_ready_ns);
            const auto lpn = first_lpn + i;
            const auto range_begin = i == 0 ? logical.offset : 0;
            const auto range_bytes = std::min(
                remaining_bytes, config_.device.page_size_bytes - range_begin);
            const auto range_end = range_begin + range_bytes;
            remaining_bytes -= range_bytes;
            const auto target_stack = physical_request ?
                stack_of_block(static_cast<std::size_t>(
                    (encode_ppn(decode(request.addr)) + i) /
                    config_.device.pages_per_block)) :
                stack_for_lpn(lpn);
            page_ready_ns = wait_for_page_stack(target_stack, page_ready_ns);
            const bool full_page_overwrite =
                range_begin == 0 && range_end == config_.device.page_size_bytes;
            const auto old_ppn = physical_request ?
                std::optional<std::uint64_t>{} :
                lookup_lpn(
                    lpn,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    full_page_overwrite ?
                        LookupIntent::OverwriteFullPage :
                        LookupIntent::ReadData);
            // Physical writes name their target page directly; GC pressure
            // belongs to that page's stack, not to stack_for_lpn of a raw
            // physical address reinterpreted as an LPN.
            if (!physical_request && config_.host.write_coalescing_enabled) {
                double page_done = stage_write_buffer_range(
                    lpn,
                    range_begin,
                    range_end,
                    request.heatmap_source,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans);
                if (config_.host.write_buffer_completion_requires_flush) {
                    flush_write_buffer_entry(lpn, page_done, out.breakdown, trace_spans);
                } else {
                    // Slot admission happens inside stage_write_buffer_range,
                    // before the new range becomes visible. Threshold flushes
                    // run on the device timeline in the background.
                    if (config_.host.write_buffer_flush_threshold_pages != 0) {
                        double background_ns = page_done;
                        const auto wb_stack = stack_for_lpn(lpn);
                        auto& die_lru = write_buffer_lru(wb_stack);
                        while (write_buffer(wb_stack).size() >=
                               config_.host.write_buffer_flush_threshold_pages) {
                            const auto victim_lpn = die_lru.back();
                            flush_write_buffer_entry(
                                victim_lpn,
                                background_ns,
                                out.breakdown,
                                trace_spans);
                        }
                        background_finish_ns_ = std::max(background_finish_ns_, background_ns);
                    }
                }
                if (first_write_path.empty()) {
                    first_write_path = "lpn" + std::to_string(first_lpn) + "->write_buffer";
                }
                finish_ns = std::max(finish_ns, page_done);
                continue;
            }
            if (!physical_request && old_ppn) {
                if (!full_page_overwrite) {
                    if (trace_spans != nullptr) {
                        add_trace_span(
                            trace_spans,
                            "partial_page_merge",
                            "translation",
                            logic_entity(ingress_stack),
                            page_ready_ns,
                            page_ready_ns + config_.host.mapping_update_ns,
                            true,
                            "lpn" + std::to_string(lpn));
                    }
                    out.breakdown.translation_ns += config_.host.mapping_update_ns;
                    page_ready_ns += config_.host.mapping_update_ns;
                    page_ready_ns = schedule_read_page(
                        *old_ppn,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans,
                        TransactionSource::User,
                        request.heatmap_source,
                        ReadPayloadRoute::HostBuffer,
                        0);
                    stats_.physical_read_bytes += config_.device.page_size_bytes;
                    stats_.page_reads++;
                }
            }
            if (!full_page_overwrite) {
                // Materialize a complete page in the reserved HBM storage.
                // Existing backing data were installed by HostBuffer above;
                // a new partial page also initializes its untouched bytes.
                page_ready_ns = host_memory_transfer(target_stack,
                    old_ppn ? range_bytes : config_.device.page_size_bytes,
                    Op::Write, page_ready_ns, out.breakdown, trace_spans);
                page_ready_ns = host_memory_transfer(target_stack, config_.device.page_size_bytes,
                    Op::Read, page_ready_ns, out.breakdown, trace_spans);
            }
            const auto ingress_detail = trace_spans == nullptr ?
                std::string{} :
                "request " + request.id + " page " + std::to_string(i);
            // Application writes arrive in host memory. The device transfer
            // occurs only after allocation, when a complete page is programmed.
            (void)ingress_detail;
            if (!physical_request) {
                maybe_run_gc(page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    1,
                    target_stack,
                    BlockRole::Data);
            }

            const auto new_ppn = physical_request ?
                encode_ppn(decode(request.addr)) + i :
                allocate_free_page(
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    BlockRole::Data,
                    stack_for_lpn(lpn));
            if (i == 0) {
                first_ppn = new_ppn;
            }
            const double program_done = schedule_program_page(
                new_ppn,
                page_ready_ns,
                out.breakdown,
                trace_spans,
                TransactionSource::User,
                request.heatmap_source);
            double mapping_ready = program_done;
            double page_done = program_done;
            if (!physical_request) {
                const auto program_commit_sequence = schedule_media_program_commit(
                    new_ppn,
                    lpn,
                    PageOwner::Logical,
                    program_done);
                access_mapping(
                    lpn,
                    TransactionSource::User,
                    MappingAccessKind::Update,
                    mapping_ready,
                    out.breakdown,
                    trace_spans);
                page_done = mapping_ready;
                schedule_lpn_mapping_commit(
                    lpn, new_ppn, page_done, program_commit_sequence);
                mark_mapping_page_dirty(
                    mapping_vpn_for_lpn(lpn), page_done);
                // The resident entry is authoritative immediately; its dirty
                // checkpoint page is persisted by the end-of-run drain.
            } else {
                schedule_physical_program_commit(new_ppn, page_done);
            }
            finish_ns = std::max(finish_ns, page_done);
        }
        out.finish_ns = finish_ns;
        out.resource_path = physical_request ? decode(request.addr).path() :
            (!first_write_path.empty() ? first_write_path :
                ("lpn" + std::to_string(first_lpn) + "->" + decode_ppn(first_ppn).path()));
        if (!physical_request && config_.host.write_coalescing_enabled) {
            out.note = pages == 1 ? "write-buffer-stage" : "multi-page-write-buffer-stage";
        } else {
            out.note = pages == 1 ? "page-program-map-update" : "multi-page-program-map-update";
        }

        stats_.program_requests++;
        if (!physical_request) {
            stats_.logical_write_bytes += request.bytes;
            if (config_.host.static_wear_leveling_max_write_fraction > 0.0) {
                // LPNs are striped across stacks; account only payload bytes,
                // including a possible partial first/last page.
                for (std::uint64_t p = 0; p < pages; ++p) {
                    const auto lpn = first_lpn + p;
                    const auto begin = std::max(request.addr, lpn * config_.device.page_size_bytes);
                    const auto end = std::min(request.addr + request.bytes,
                        (lpn + 1) * config_.device.page_size_bytes);
                    auto& credit = wear_leveling_credit_by_stack_.at(stack_for_lpn(lpn));
                    credit = std::min(
                        static_cast<double>(config_.device.pages_per_block) * config_.device.page_size_bytes,
                        credit + (end - begin) * config_.host.static_wear_leveling_max_write_fraction);
                }
            }
        }
        if (physical_request || !config_.host.write_coalescing_enabled) {
            const auto payload_bytes = pages * config_.device.page_size_bytes;
            stats_.physical_write_bytes += payload_bytes;
            stats_.data_program_payload_bytes += payload_bytes;
            stats_.data_programs += pages;
            stats_.page_programs += pages;
            if (physical_request) {
                stats_.raw_physical_program_payload_bytes += payload_bytes;
                stats_.raw_physical_programs += pages;
            }
        }
    } else if (request.op == Op::Erase) {
        const auto addr = decode(request.addr);
        const auto block = block_index(addr);
        out.finish_ns = schedule_erase_block(
            block,
            issued_ns,
            out.breakdown,
            trace_spans,
            TransactionSource::User,
            request.heatmap_source);
        out.resource_path = addr.path();
        out.note = "physical-block-erase";
        out.physical_bytes = 0;
        const auto [pending_erase, inserted] = pending_block_transitions_.emplace(
            block,
            PendingBlockTransition{.finish_ns = out.finish_ns});
        if (!inserted) {
            throw std::runtime_error("HBF block already has an in-flight physical erase");
        }
        auto& block_state = blocks_.at(block);
        if (block_state.epoch == std::numeric_limits<std::uint64_t>::max()) {
            pending_block_transitions_.erase(block);
            throw std::runtime_error("HBF block epoch overflow");
        }
        // A raw erase may legally target an already-erased block. Such a
        // block is still present in the plane's free pool, so retire it from
        // allocation before publishing erase_pending. Otherwise a same-time
        // logical write can claim the block while its erase completion is
        // already scheduled; that completion then destroys the new page.
        auto& erased_plane = planes_.at(block_plane_index(block));
        mark_gc_headroom_dirty(block_plane_index(block));
        if (block_state.role == BlockRole::Free) {
            const auto free_block = std::find(
                erased_plane.free_blocks.begin(),
                erased_plane.free_blocks.end(),
                block);
            if (free_block == erased_plane.free_blocks.end()) {
                pending_block_transitions_.erase(block);
                throw std::runtime_error(
                    "HBF free block targeted by erase is missing from its plane pool");
            }
            erased_plane.free_blocks.erase(free_block);
        }
        const auto retired_block_epoch = block_state.epoch;
        tag_pending_mapping_updates_for_erase(
            block, retired_block_epoch, out.finish_ns);
        // Capacity enforcement no longer scans every cache line looking for
        // stale epochs.  Retire this block's decoded lines at the same point
        // that the block becomes erase-pending; this is the visibility point
        // used by the former scan as well.
        read_buffer_purge_block(block);
        device_dram_purge_block(block);
        block_state.epoch++;
        block_state.erase_pending = true;
        const auto erase_epoch = block_state.epoch;
        if (erased_plane.active_data_block == block) {
            erased_plane.active_data_block = std::nullopt;
        }
        if (erased_plane.active_mapping_block == block) {
            erased_plane.active_mapping_block = std::nullopt;
        }
        if (erased_plane.active_gc_block == block) {
            erased_plane.active_gc_block = std::nullopt;
        }

        pending_erase->second.commit_sequence = schedule_commit(
            addr.stack, out.finish_ns, [this, block, erase_epoch]() {
            const auto pending_erase = pending_block_transitions_.find(block);
            if (pending_erase == pending_block_transitions_.end()) {
                throw std::runtime_error("HBF physical erase lost its in-flight reservation");
            }
            if (blocks_.at(block).epoch != erase_epoch ||
                !blocks_.at(block).erase_pending) {
                throw std::runtime_error(
                    "HBF physical erase lost its block-epoch ownership");
            }
            const auto block_begin = static_cast<std::uint64_t>(
                block) * config_.device.pages_per_block;
            for (std::uint32_t page = 0; page < config_.device.pages_per_block; ++page) {
                const auto ppn = block_begin + page;
                const auto it = programmed_pages_.find(ppn);
                if (it == programmed_pages_.end()) {
                    continue;
                }
                if (it->second.status == PageStatus::Valid) {
                    if (it->second.owner == PageOwner::Mapping) {
                        const auto mapping_vpn = metadata_vpn(it->second.lpn);
                        materialized_ready_by_vpn_[mapping_vpn] = std::max(
                            materialized_ready_by_vpn_[mapping_vpn],
                            pending_erase->second.finish_ns);
                        const auto found =
                            mapping_vpn_to_ppn_.find(mapping_vpn);
                        if (found != mapping_vpn_to_ppn_.end() && found->second == ppn) {
                            mapping_vpn_to_ppn_.erase(found);
                        }
                    } else if (it->second.owner == PageOwner::Logical) {
                        materialized_ready_by_lpn_[it->second.lpn] = std::max(
                            materialized_ready_by_lpn_[it->second.lpn],
                            pending_erase->second.finish_ns);
                        const auto found = lpn_to_ppn_.find(it->second.lpn);
                        if (found != lpn_to_ppn_.end() && found->second == ppn) {
                            lpn_to_ppn_.erase(found);
                        }
                    }
                }
            }
            release_invalid_block(block);
            retire_mapping_update_tombstones(
                block, pending_erase->second.finish_ns);
            materialized_ready_by_block_.at(block) = std::max(
                materialized_ready_by_block_.at(block),
                pending_erase->second.finish_ns);
            pending_block_transitions_.erase(pending_erase);
        });

        stats_.erase_requests++;
    }

    if (address_heatmap_ != nullptr && !physical_request &&
        (request.op == Op::Read || request.op == Op::Write)) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::HbfLogical,
            .direction = request.op == Op::Read ?
                TrafficDirection::Read : TrafficDirection::Write,
            .source = request.heatmap_source,
            .address = request.addr,
            .bytes = request.bytes,
        });
    }
    stats_.free_pages = free_pages_;
    stats_.stage_work += out.breakdown;
    stats_.finish_ns = std::max({stats_.finish_ns, out.finish_ns, background_finish_ns_});
    if (request.op == Op::Read || request.op == Op::Write) {
        out.physical_bytes = stats_.physical_read_bytes + stats_.physical_write_bytes -
            physical_bytes_before;
    }
    return out;
}

std::size_t HbfController::stack_index(const HbfAddress& addr) const {
    return addr.stack;
}

std::size_t HbfController::channel_index(const HbfAddress& addr) const {
    return static_cast<std::size_t>(addr.stack) * config_.device.channels_per_stack + addr.channel;
}

std::size_t HbfController::die_index(const HbfAddress& addr) const {
    std::size_t index = addr.stack;
    index = index * config_.device.channels_per_stack + addr.channel;
    index = index * config_.device.dies_per_channel + addr.die;
    return index;
}

std::list<std::uint64_t>& HbfController::write_buffer_lru(std::size_t stack) {
    return write_buffer_lru_by_stack_.at(stack);
}

std::unordered_map<std::uint64_t, HbfController::WriteBufferEntry>& HbfController::write_buffer(
    std::size_t stack) {
    return write_buffer_by_stack_.at(stack);
}


double HbfController::preview_full_plane_window(
    PlaneState& plane,
    double earliest_ns,
    double duration_ns) {
    plane.array_barrier.prune_before(reservation_causal_watermark_ns_);
    for (auto& subarray : plane.subarrays) {
        subarray.timeline.prune_before(reservation_causal_watermark_ns_);
    }
    for (auto& lane : plane.media_lanes) {
        lane.timeline.prune_before(reservation_causal_watermark_ns_);
    }
    for (auto& bank : plane.page_buffer_banks) {
        bank.timeline.prune_before(reservation_causal_watermark_ns_);
    }

    double candidate = earliest_ns;
    for (;;) {
        double next = plane.array_barrier.preview_start(
            candidate, duration_ns);
        for (const auto& subarray : plane.subarrays) {
            next = std::max(
                next, subarray.timeline.preview_start(candidate, duration_ns));
        }
        for (const auto& lane : plane.media_lanes) {
            next = std::max(
                next, lane.timeline.preview_start(candidate, duration_ns));
        }
        for (const auto& bank : plane.page_buffer_banks) {
            next = std::max(
                next, bank.timeline.preview_start(candidate, duration_ns));
        }

        bool exact = plane.array_barrier.can_reserve_exact(
            next, duration_ns);
        for (const auto& subarray : plane.subarrays) {
            exact = exact && subarray.timeline.can_reserve_exact(next, duration_ns);
        }
        for (const auto& lane : plane.media_lanes) {
            exact = exact && lane.timeline.can_reserve_exact(next, duration_ns);
        }
        for (const auto& bank : plane.page_buffer_banks) {
            exact = exact && bank.timeline.can_reserve_exact(next, duration_ns);
        }
        if (exact) {
            return next;
        }
        if (next <= candidate) {
            throw std::runtime_error(
                "HBF full-plane preview failed to make forward progress");
        }
        candidate = next;
    }
}

void HbfController::record_full_plane_window(
    PlaneState& plane,
    double begin_ns,
    double end_ns) {
    // Windows are disjoint and sorted, so the ones that end at or before
    // the causal watermark form a prefix that no future read can straddle.
    // Drop it once it dominates the vector (amortized O(1)).
    {
        auto& windows = plane.full_plane_windows;
        const auto dead_end = std::upper_bound(
            windows.begin(),
            windows.end(),
            reservation_causal_watermark_ns_,
            [](double watermark, const PlaneState::BusyWindow& window) {
                return watermark < window.end_ns;
            });
        const auto dead = static_cast<std::size_t>(dead_end - windows.begin());
        if (dead >= 64 && dead * 2 >= windows.size()) {
            windows.erase(windows.begin(), dead_end);
        }
    }
    const auto position = std::lower_bound(
        plane.full_plane_windows.begin(),
        plane.full_plane_windows.end(),
        begin_ns,
        [](const PlaneState::BusyWindow& window, double begin) {
            return window.begin_ns < begin;
        });
    if ((position != plane.full_plane_windows.begin() &&
         std::prev(position)->end_ns > begin_ns) ||
        (position != plane.full_plane_windows.end() &&
         position->begin_ns < end_ns)) {
        throw std::runtime_error("HBF full-plane windows overlap");
    }
    plane.full_plane_windows.insert(position, PlaneState::BusyWindow{
        .begin_ns = begin_ns,
        .end_ns = end_ns,
    });
}

void HbfController::fold_ecc_inflight_intervals(
    DieState& die,
    double causal_arrival_watermark_ns) {
    if (die.ecc_inflight_intervals.empty()) {
        die.ecc_folded_through_ns = std::max(
            die.ecc_folded_through_ns, causal_arrival_watermark_ns);
        die.ecc_intervals_at_last_fold = 0;
        return;
    }
    // Sweep the retained intervals. Every interval alive at any time point
    // in (folded_through, watermark] is retained, because pruned intervals
    // all finished at or before the previous fold; the level at each event
    // time up to the watermark is therefore exact.
    std::vector<std::pair<double, int>> events;
    events.reserve(die.ecc_inflight_intervals.size() * 2);
    for (const auto& interval : die.ecc_inflight_intervals) {
        events.emplace_back(interval.start_ns, 1);
        events.emplace_back(interval.finish_ns, -1);
    }
    std::sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    });
    std::int64_t inflight = 0;
    for (const auto& [time_ns, delta] : events) {
        if (time_ns > causal_arrival_watermark_ns) {
            break;
        }
        inflight += delta;
        if (inflight < 0) {
            throw std::runtime_error(
                "HBF ECC in-flight accounting became negative while folding");
        }
        die.ecc_max_inflight_folded = std::max(
            die.ecc_max_inflight_folded, static_cast<std::uint64_t>(inflight));
    }
    std::erase_if(
        die.ecc_inflight_intervals,
        [causal_arrival_watermark_ns](const DieState::EccInflightInterval& interval) {
            return interval.finish_ns <= causal_arrival_watermark_ns;
        });
    die.ecc_folded_through_ns = std::max(
        die.ecc_folded_through_ns, causal_arrival_watermark_ns);
    die.ecc_intervals_at_last_fold = die.ecc_inflight_intervals.size();
}

std::uint64_t HbfController::ecc_max_inflight(const DieState& die) {
    // Retained intervals alone undercount the level at time points behind
    // the fold, but those points are covered by the folded peak, so the
    // maximum of the two is the exact all-time peak.
    std::vector<std::pair<double, int>> events;
    events.reserve(die.ecc_inflight_intervals.size() * 2);
    for (const auto& interval : die.ecc_inflight_intervals) {
        events.emplace_back(interval.start_ns, 1);
        events.emplace_back(interval.finish_ns, -1);
    }
    std::sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        // A completion at t leaves before a new codeword enters at t.
        return lhs.second < rhs.second;
    });
    std::int64_t inflight = 0;
    std::uint64_t max_inflight = die.ecc_max_inflight_folded;
    for (const auto& [_, delta] : events) {
        inflight += delta;
        if (inflight < 0) {
            throw std::runtime_error(
                "HBF ECC in-flight accounting became negative");
        }
        max_inflight = std::max(
            max_inflight, static_cast<std::uint64_t>(inflight));
    }
    if (inflight != 0) {
        throw std::runtime_error(
            "HBF ECC in-flight accounting did not return to zero");
    }
    return max_inflight;
}

void HbfController::prune_expired_state(double causal_arrival_watermark_ns) {
    // ECC latency intervals: fold once a die has doubled its retained set
    // since the last fold and holds more than a small floor, so the sweep
    // cost is amortized over the codewords it retires.
    for (auto& die : dies_) {
        const auto retained = die.ecc_inflight_intervals.size();
        if (retained > 4096 && retained > 2 * die.ecc_intervals_at_last_fold) {
            fold_ecc_inflight_intervals(die, causal_arrival_watermark_ns);
        }
    }
    // Commit frontiers: every reader compares them against a time at or
    // after the watermark, so entries at or behind it are dead.
    const auto materialized_size = materialized_ready_by_lpn_.size() +
        materialized_ready_by_vpn_.size() + materialized_ready_by_ppn_.size();
    if (materialized_size > 1024 &&
        materialized_size > 2 * materialized_ready_prune_baseline_) {
        const auto expired = [causal_arrival_watermark_ns](const auto& entry) {
            return entry.second <= causal_arrival_watermark_ns;
        };
        std::erase_if(materialized_ready_by_lpn_, expired);
        std::erase_if(materialized_ready_by_vpn_, expired);
        std::erase_if(materialized_ready_by_ppn_, expired);
        materialized_ready_prune_baseline_ = materialized_ready_by_lpn_.size() +
            materialized_ready_by_vpn_.size() +
            materialized_ready_by_ppn_.size();
    }
}

void HbfController::advance_hbf_frontier(double lower_bound_ns) {
    // Deferred publications can still observe a mapping version at their own
    // timestamp. Do not fold that history until the callback has been applied.
    if (!pending_commits_.empty())
        lower_bound_ns = std::min(lower_bound_ns, pending_commits_.begin()->first.first);
    reservation_causal_watermark_ns_ = std::max(
        reservation_causal_watermark_ns_, lower_bound_ns);
    prune_expired_state(reservation_causal_watermark_ns_);
    media_->advance_cache(reservation_causal_watermark_ns_);
}

void HbfController::advance_administrative_frontier(double lower_bound_ns) {
    advance_hbf_frontier(lower_bound_ns);
    if (buffer_hbm_)
        buffer_hbm_->advance_buffer_frontier(reservation_causal_watermark_ns_);
}

void HbfController::record_plane_media_busy(
    PlaneState& plane,
    double begin_ns,
    double end_ns) {
    if (end_ns <= begin_ns) {
        return;
    }
    auto& windows = plane.media_busy_windows;
    // The union is sorted and disjoint; windows that end at or before the
    // causal watermark can never merge with a future reservation (every
    // future reservation starts at or after the watermark) and their busy
    // time is already accumulated. Drop that dead prefix once it dominates
    // the vector (amortized O(1)); the union total is unaffected.
    {
        const auto dead_end = std::upper_bound(
            windows.begin(),
            windows.end(),
            reservation_causal_watermark_ns_,
            [](double watermark, const PlaneState::BusyWindow& window) {
                return watermark < window.end_ns;
            });
        const auto dead = static_cast<std::size_t>(dead_end - windows.begin());
        if (dead >= 64 && dead * 2 >= windows.size()) {
            windows.erase(windows.begin(), dead_end);
        }
    }
    auto first = std::lower_bound(
        windows.begin(),
        windows.end(),
        begin_ns,
        [](const PlaneState::BusyWindow& window, double begin) {
            return window.end_ns < begin;
        });
    double merged_begin = begin_ns;
    double merged_end = end_ns;
    double replaced_ns = 0.0;
    auto last = first;
    while (last != windows.end() && last->begin_ns <= merged_end) {
        merged_begin = std::min(merged_begin, last->begin_ns);
        merged_end = std::max(merged_end, last->end_ns);
        replaced_ns += last->end_ns - last->begin_ns;
        ++last;
    }
    first = windows.erase(first, last);
    windows.insert(first, PlaneState::BusyWindow{
        .begin_ns = merged_begin,
        .end_ns = merged_end,
    });
    plane.media_busy_ns += merged_end - merged_begin - replaced_ns;
}

bool HbfController::read_buffer_contains(std::uint64_t ppn, double at_ns) {
    const auto& block = blocks_.at(ppn / config_.device.pages_per_block);
    return !block.erase_pending && media_->cache_hit(ppn, at_ns);
}
void HbfController::read_buffer_insert(std::uint64_t ppn, double ready_ns) {
    if (!blocks_.at(ppn / config_.device.pages_per_block).erase_pending)
        media_->cache_fill(ppn, ready_ns);
}
void HbfController::read_buffer_purge_page(std::uint64_t ppn) { media_->cache_purge_page(ppn); }
void HbfController::read_buffer_purge_block(std::size_t block) { media_->cache_purge_block(block); }

double HbfController::serve_read_from_read_buffer(
    std::uint64_t ppn,
    std::uint64_t bytes,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans, bool host_buffer) {
    const auto stack = static_cast<std::uint32_t>(stack_of_block(
        static_cast<std::size_t>(ppn / config_.device.pages_per_block)));
    auto& logic_die = logic_dies_.at(stack);
    if (!mapping_is_local())
        earliest_ns = schedule_external_request_command(channel_index(decode_ppn(ppn)),
            earliest_ns, breakdown, spans, "cached page read", "user");
    if (bytes == 0 || bytes > config_.device.page_size_bytes) {
        throw std::runtime_error("HBF read-buffer transfer size is invalid");
    }
    const double sram_ns = transfer_time_ns(bytes, config_.device.logic_sram_bandwidth_GBps);
    auto sram = reserve(earliest_ns, sram_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    if (spans != nullptr) {
        trace_wait(
            spans, logic_entity(stack), earliest_ns, sram.start_ns, "wait_sram");
        add_trace_span(
            spans,
            "read_buffer_hit",
            "sram",
            logic_entity(stack),
            sram.start_ns,
            sram.finish_ns,
            true,
            "ppn" + std::to_string(ppn));
    }
    stats_.read_buffer_hits++;
    stats_.read_buffer_read_bytes += bytes;
    // The hit leaves over the external interface as requested decoded bytes;
    // OOB never appears beyond ECC/SRAM.
    const auto response = schedule_external_read_egress(
        channel_index(decode_ppn(ppn)),
        bytes,
        sram.finish_ns,
        breakdown,
        spans,
        "read_buffer_hit_hbio_out",
        "read-buffer hit");
    blocks_.at(ppn / config_.device.pages_per_block).issued_media_ready_ns = std::max(
        blocks_.at(ppn / config_.device.pages_per_block).issued_media_ready_ns, response);
    return host_buffer ? host_memory_transfer(stack, bytes, Op::Write,
        response, breakdown, spans) : response;
}

double HbfController::device_dram_transfer(std::size_t stack, std::uint64_t bytes,
    bool write, double earliest_ns, Breakdown& breakdown,
    std::vector<TraceSpan>* spans, const char* name) {
    if (bytes == 0) return earliest_ns;
    // The CXL-SSD device-DRAM envelope: the access latency precedes the
    // port occupancy of bytes / bandwidth.
    const double access_ns = earliest_ns + config_.host.device_dram_hit_latency_ns;
    const double transfer_ns = transfer_time_ns(bytes, config_.host.device_dram_bandwidth_GBps);
    const auto slot = reserve(access_ns, transfer_ns, device_dram_port_by_stack_.at(stack));
    breakdown.scheduler_queue_wait_ns += slot.wait_ns;
    breakdown.write_buffer_dram_ns += config_.host.device_dram_hit_latency_ns + transfer_ns;
    if (write) {
        stats_.device_dram_write_bytes = checked_add(
            stats_.device_dram_write_bytes, bytes, "HBF device DRAM write bytes");
    } else {
        stats_.device_dram_read_bytes = checked_add(
            stats_.device_dram_read_bytes, bytes, "HBF device DRAM read bytes");
    }
    if (spans != nullptr) {
        const auto entity = logic_entity(static_cast<std::uint32_t>(stack)) + "/device_dram";
        trace_wait(spans, entity, access_ns, slot.start_ns, "wait_device_dram");
        add_trace_span(spans, name, "device_dram", entity, earliest_ns, slot.finish_ns,
            true, std::to_string(bytes) + "B");
    }
    return slot.finish_ns;
}

std::optional<double> HbfController::device_dram_probe(std::uint64_t ppn) {
    const auto stack = stack_of_block(static_cast<std::size_t>(ppn / config_.device.pages_per_block));
    auto& lines = device_dram_by_stack_.at(stack);
    const auto found = lines.find(ppn);
    if (found == lines.end()) {
        ++stats_.device_dram_read_misses;
        return std::nullopt;
    }
    ++stats_.device_dram_read_hits;
    if (config_.host.device_dram_policy == DeviceDramPolicy::Lru) {
        auto& order = device_dram_order_by_stack_.at(stack);
        order.splice(order.begin(), order, found->second.iterator);
    }
    return found->second.ready_ns;
}

std::optional<double> HbfController::device_dram_cached(std::uint64_t ppn) const {
    const auto stack = stack_of_block(static_cast<std::size_t>(ppn / config_.device.pages_per_block));
    const auto& lines = device_dram_by_stack_.at(stack);
    const auto found = lines.find(ppn);
    if (found == lines.end()) return std::nullopt;
    return found->second.ready_ns;
}

double HbfController::device_dram_serve_hit(std::uint64_t ppn, std::uint64_t bytes,
    double earliest_ns, double line_ready_ns, Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto channel = channel_index(decode_ppn(ppn));
    const auto stack = channel / config_.device.channels_per_stack;
    if (!mapping_is_local())
        earliest_ns = schedule_external_request_command(channel, earliest_ns, breakdown, spans,
            "device-DRAM page read", "user");
    if (line_ready_ns > earliest_ns) {
        breakdown.scheduler_queue_wait_ns += line_ready_ns - earliest_ns;
        if (spans != nullptr)
            trace_wait(spans, logic_entity(static_cast<std::uint32_t>(stack)) + "/device_dram",
                earliest_ns, line_ready_ns, "wait_device_dram_fill");
        earliest_ns = line_ready_ns;
    }
    const auto read = device_dram_transfer(stack, bytes, false, earliest_ns, breakdown, spans,
        "device_dram_read_hit");
    stats_.device_dram_read_hit_bytes += bytes;
    return schedule_external_read_egress(channel, bytes, read, breakdown, spans,
        "device_dram_hit_hbio_out", "device-DRAM hit");
}

void HbfController::device_dram_fill(std::uint64_t ppn, double data_ready_ns,
    std::vector<TraceSpan>* spans) {
    const auto block = static_cast<std::size_t>(ppn / config_.device.pages_per_block);
    const auto stack = stack_of_block(block);
    // A block already scheduled for erase must not regain cached lines, and a
    // page that is already cached needs no second copy.
    if (blocks_.at(block).erase_pending || device_dram_by_stack_.at(stack).contains(ppn)) return;
    // The fill starts when the decoded page is ready and takes the port's
    // access latency plus the page transfer. It is background work: it is
    // admitted only into idle time of the host-facing port calendar and of
    // the fill engine (so fills never queue behind each other) and yields to
    // every later host-facing access, which is why it is never reserved on
    // the port calendar and never charged to the reading request.
    const double access_ns = data_ready_ns + config_.host.device_dram_hit_latency_ns;
    const double transfer_ns = transfer_time_ns(config_.device.page_size_bytes,
        config_.host.device_dram_bandwidth_GBps);
    auto& port = device_dram_port_by_stack_.at(stack);
    auto& engine = device_dram_fill_by_stack_.at(stack);
    port.prune_before(reservation_causal_watermark_ns_);
    engine.prune_before(reservation_causal_watermark_ns_);
    if (!port.can_reserve_exact(access_ns, transfer_ns) ||
        !engine.can_reserve_exact(access_ns, transfer_ns)) {
        ++stats_.device_dram_fill_bypasses;
        return;
    }
    const auto slot = reserve(access_ns, transfer_ns, engine);
    if (slot.wait_ns != 0.0)
        throw std::logic_error("HBF device DRAM fill queued despite an idle fill engine");
    stats_.device_dram_write_bytes = checked_add(stats_.device_dram_write_bytes,
        config_.device.page_size_bytes, "HBF device DRAM write bytes");
    if (spans != nullptr)
        add_trace_span(spans, "device_dram_fill", "device_dram",
            logic_entity(static_cast<std::uint32_t>(stack)) + "/device_dram_fill",
            data_ready_ns, slot.finish_ns, false, "ppn" + std::to_string(ppn));
    device_dram_install(ppn, slot.finish_ns);
}

void HbfController::device_dram_install(std::uint64_t ppn, double ready_ns) {
    const auto block = static_cast<std::size_t>(ppn / config_.device.pages_per_block);
    const auto stack = stack_of_block(block);
    // A block already scheduled for erase must not regain cached lines.
    if (blocks_.at(block).erase_pending) return;
    auto& lines = device_dram_by_stack_.at(stack);
    auto& order = device_dram_order_by_stack_.at(stack);
    if (const auto found = lines.find(ppn); found != lines.end()) {
        found->second.ready_ns = std::min(found->second.ready_ns, ready_ns);
        return;
    }
    if (lines.size() >= device_dram_.read_cache_pages_per_stack) {
        lines.erase(order.back());
        order.pop_back();
        ++stats_.device_dram_evictions;
    }
    order.push_front(ppn);
    lines.emplace(ppn, DeviceDramLine{.ready_ns = ready_ns, .iterator = order.begin()});
    ++stats_.device_dram_fills;
}

void HbfController::device_dram_purge_page(std::uint64_t ppn) {
    if (!device_dram_enabled()) return;
    const auto stack = stack_of_block(static_cast<std::size_t>(ppn / config_.device.pages_per_block));
    auto& lines = device_dram_by_stack_.at(stack);
    const auto found = lines.find(ppn);
    if (found == lines.end()) return;
    device_dram_order_by_stack_.at(stack).erase(found->second.iterator);
    lines.erase(found);
}

void HbfController::device_dram_purge_block(std::size_t block_index) {
    if (!device_dram_enabled() || device_dram_by_stack_.at(stack_of_block(block_index)).empty()) return;
    const auto first = static_cast<std::uint64_t>(block_index) * config_.device.pages_per_block;
    for (std::uint64_t page = 0; page < config_.device.pages_per_block; ++page)
        device_dram_purge_page(first + page);
}

double HbfController::device_dram_ingress(std::size_t stack, std::uint64_t key,
    std::uint64_t bytes, double earliest_ns, Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto channel = stack * config_.device.channels_per_stack +
        key % config_.device.channels_per_stack;
    const auto command = schedule_external_request_command(channel, earliest_ns, breakdown, spans,
        "device-DRAM write", "user");
    return schedule_external_write_ingress(channel, bytes, command, breakdown, spans,
        "device-DRAM write buffer", "ocp_write_buffer_ingress");
}

double HbfController::device_dram_stage_payload(std::size_t stack, std::uint64_t key,
    std::uint64_t bytes, double earliest_ns, Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    return device_dram_transfer(stack, bytes, true,
        device_dram_ingress(stack, key, bytes, earliest_ns, breakdown, spans),
        breakdown, spans, "write_buffer_stage_device_dram");
}

double HbfController::device_dram_respond(std::size_t stack, std::uint64_t key,
    std::uint64_t bytes, double earliest_ns, Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto channel = stack * config_.device.channels_per_stack +
        key % config_.device.channels_per_stack;
    const auto command = schedule_external_request_command(channel, earliest_ns, breakdown, spans,
        "device-DRAM read", "user");
    const auto read = device_dram_transfer(stack, bytes, false, command, breakdown, spans,
        "write_buffer_read_device_dram");
    return schedule_external_read_egress(channel, bytes, read, breakdown, spans,
        "device_dram_read_hbio_out", "device-DRAM payload");
}

std::size_t HbfController::stack_for_lpn(std::uint64_t lpn) const {
    // Page-granular striping keeps a sequential window active on every HBF
    // stack. mapping_vpn_for_lpn() still groups each stack's local sequence
    // into its own mapping pages, so metadata never needs a cross-stack
    // lookup. This deliberately separates data striping from metadata
    // ownership; binding 512 consecutive global pages to one mapping page
    // serialized an entire W512 stream onto one stack.
    return stack_for_vpn(mapping_vpn_for_lpn(lpn));
}

std::size_t HbfController::stack_for_vpn(std::uint64_t mapping_vpn) const {
    return static_cast<std::size_t>(mapping_vpn % config_.device.stacks);
}

std::size_t HbfController::planes_per_stack() const {
    return planes_.size() / config_.device.stacks;
}

std::size_t HbfController::stack_of_plane(std::size_t plane) const {
    return plane / planes_per_stack();
}

std::size_t HbfController::stack_of_block(std::size_t block_index) const {
    return stack_of_plane(block_plane_index(block_index));
}

std::size_t HbfController::mapping_plane_for_vpn(std::uint64_t mapping_vpn) const {
    // A VPN is encoded as group * stacks + owner_stack. Divide the owner out
    // before selecting a local plane so every stack uses all of its planes.
    return stack_for_vpn(mapping_vpn) * planes_per_stack() +
        static_cast<std::size_t>(
            (mapping_vpn / config_.device.stacks) % planes_per_stack());
}

std::size_t HbfController::plane_index(const HbfAddress& addr) const {
    std::size_t index = die_index(addr);
    index = index * config_.device.planes_per_die + addr.plane;
    return index;
}

std::size_t HbfController::block_index(const HbfAddress& addr) const {
    std::size_t index = plane_index(addr);
    index = index * config_.device.blocks_per_plane + addr.block;
    return index;
}

std::size_t HbfController::block_plane_index(std::size_t block_index) const {
    return block_index / config_.device.blocks_per_plane;
}

std::size_t HbfController::subarray_index(const HbfAddress& addr) const {
    const auto page_in_plane =
        static_cast<std::uint64_t>(addr.block) * config_.device.pages_per_block + addr.page;
    return static_cast<std::size_t>(page_in_plane % subarrays_per_plane_);
}

std::size_t HbfController::media_lane_index(const HbfAddress& addr) const {
    const auto page_in_plane =
        static_cast<std::uint64_t>(addr.block) * config_.device.pages_per_block + addr.page;
    return static_cast<std::size_t>(page_in_plane % config_.device.media_lanes_per_plane);
}

std::size_t HbfController::page_buffer_bank_index(const HbfAddress& addr) const {
    const auto page_in_plane =
        static_cast<std::uint64_t>(addr.block) * config_.device.pages_per_block + addr.page;
    return static_cast<std::size_t>(page_in_plane % config_.device.page_buffer_banks_per_plane);
}

std::uint64_t HbfController::page_count_for(const HbfAddress& addr, std::uint64_t bytes) const {
    if (bytes == 0) {
        return 0;
    }
    const auto first_page_bytes = config_.device.page_size_bytes - addr.offset;
    if (bytes <= first_page_bytes) {
        return 1;
    }
    return 1 + div_ceil(bytes - first_page_bytes, config_.device.page_size_bytes);
}

HbfAddress HbfController::decode_ppn(std::uint64_t ppn) const {
    if (ppn >= total_pages_) {
        throw std::runtime_error("HBF PPN is out of range");
    }
    return decode(ppn * config_.device.page_size_bytes);
}

std::uint64_t HbfController::encode_ppn(const HbfAddress& addr) const {
    if (addr.stack >= config_.device.stacks || addr.channel >= config_.device.channels_per_stack ||
        addr.die >= config_.device.dies_per_channel || addr.plane >= config_.device.planes_per_die ||
        addr.block >= config_.device.blocks_per_plane || addr.page >= config_.device.pages_per_block ||
        addr.offset >= config_.device.page_size_bytes) {
        throw std::runtime_error("HBF address field out of range");
    }
    std::uint64_t unit = addr.stack;
    unit = unit * config_.device.channels_per_stack + addr.channel;
    unit = unit * config_.device.dies_per_channel + addr.die;
    unit = unit * config_.device.planes_per_die + addr.plane;
    unit = unit * config_.device.blocks_per_plane + addr.block;
    unit = unit * config_.device.pages_per_block + addr.page;
    return unit;
}

HbfAddress HbfController::logical_page_address(std::uint64_t logical_byte_addr) const {
    HbfAddress addr;
    addr.offset = logical_byte_addr % config_.device.page_size_bytes;
    return addr;
}

std::uint64_t HbfController::mapping_vpn_for_lpn(std::uint64_t lpn) const {
    // One global stripe contains exactly one page per stack. Rotate the lane
    // assignment once per mapping group to prevent a power-of-two logical
    // stride from pinning a tensor/KV phase to one stack, then encode the
    // owner directly in the low VPN digit:
    //
    //   vpn = local_mapping_group * stacks + owner_stack
    //
    // Each VPN therefore owns mapping_entries_per_page consecutive entries
    // from one stack's local page sequence while adjacent global LPNs remain
    // page-striped across the full fabric.
    const auto stacks = static_cast<std::uint64_t>(config_.device.stacks);
    const auto stripe = lpn / stacks;
    const auto lane = lpn % stacks;
    const auto group = stripe / config_.host.mapping_entries_per_page;
    const auto rotation = placement_mix64(group) % stacks;
    const auto stack = lane >= stacks - rotation ?
        lane - (stacks - rotation) :
        lane + rotation;
    return checked_add(
        checked_mul(group, stacks, "HBF mapping VPN group"),
        stack,
        "HBF mapping VPN");
}

HbfController::CompactReadAddress HbfController::compact_lpn_address(
    std::uint64_t lpn) const {
    const auto& image = *compact_logical_image_;
    const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
    const auto stack = stack_for_vpn(mapping_vpn);
    if (mapping_vpn < image.first_vpn ||
        mapping_vpn - image.first_vpn >= image.vpn_slot_count) {
        throw std::runtime_error(
            "HBF compact LPN maps outside its VPN directory");
    }
    const auto vpn_offset = mapping_vpn - image.first_vpn;
    const auto& range = image.vpn_ranges.at(
        static_cast<std::size_t>(vpn_offset));
    const auto local_entry =
        (lpn / config_.device.stacks) % config_.host.mapping_entries_per_page;
    if (local_entry < range.first_entry ||
        local_entry - range.first_entry >= range.page_count) {
        throw std::runtime_error(
            "HBF compact LPN is outside its mapping-page range");
    }
    const auto data_index = checked_add(
        range.stack_page_offset,
        local_entry - range.first_entry,
        "HBF compact per-stack data index");
    const auto pps = static_cast<std::uint64_t>(planes_per_stack());
    const auto plane = stack * planes_per_stack() +
        static_cast<std::size_t>(data_index % pps);
    const auto page_ordinal = data_index / pps;
    const auto block_ordinal = page_ordinal / config_.device.pages_per_block;
    const auto page = static_cast<std::uint32_t>(
        page_ordinal % config_.device.pages_per_block);
    const auto& assigned_blocks = image.data_blocks_by_plane.at(plane);
    if (block_ordinal >= assigned_blocks.size()) {
        throw std::runtime_error(
            "HBF compact LPN maps beyond its data-block directory");
    }
    const auto block_index = assigned_blocks[block_ordinal];
    return CompactReadAddress{
        .ppn = block_index * config_.device.pages_per_block + page,
        .mapping_vpn = mapping_vpn,
        .stack = stack,
        .block_index = static_cast<std::size_t>(block_index),
        .page = page,
    };
}

std::optional<HbfController::CompactReadAddress> HbfController::compact_read_address(
    std::uint64_t lpn, std::uint64_t request_end_lpn, CompactReadRun& run) {
    if (lpn >= run.end_lpn) {
        run.first_lpn = lpn;
        run.end_lpn = request_end_lpn;
        run.lanes.clear();
        const auto& image = *compact_logical_image_;
        if (lpn < image.first_lpn) {
            run.end_lpn = std::min(run.end_lpn, image.first_lpn);
            return std::nullopt;
        }
        if (lpn - image.first_lpn >= image.page_count) return std::nullopt;
        run.end_lpn = std::min(run.end_lpn, image.first_lpn + image.page_count);

        // Writes/TRIM/retired GC copies split the clean address range. A GC
        // publication during a later mapping access can still retire a page:
        // compact_lpn_ppn rechecks that sparse state on every use below.
        const auto next_mutation = mutated_lpn_ranges_.upper_bound(lpn);
        if (next_mutation != mutated_lpn_ranges_.begin()) {
            const auto previous = std::prev(next_mutation);
            if (lpn < previous->second) {
                run.end_lpn = std::min(run.end_lpn, previous->second);
                return std::nullopt;
            }
        }
        if (next_mutation != mutated_lpn_ranges_.end())
            run.end_lpn = std::min(run.end_lpn, next_mutation->first);

        // Rotation and each stack's compact allocation offset stay fixed
        // within one mapping group. The bank directories are immutable after
        // prepopulation/restore, but their live bits are deliberately uncached.
        const auto group_width = checked_mul(config_.device.stacks,
            config_.host.mapping_entries_per_page, "HBF compact read group");
        run.end_lpn = lpn + std::min(run.end_lpn - lpn,
            group_width - lpn % group_width);
        run.lane_width = static_cast<std::uint64_t>(planes_.size());
        const auto lane_count = std::min(run.lane_width, run.end_lpn - lpn);
        run.lanes.reserve(static_cast<std::size_t>(lane_count));
        for (std::uint64_t lane = 0; lane < lane_count; ++lane) {
            auto address = compact_lpn_address(lpn + lane);
            const auto pages_left = config_.device.pages_per_block - address.page;
            // Stop before any lane rolls into another physical block. That
            // next block may not be adjacent to this one in the PPN space.
            const auto lane_pages = (run.end_lpn - (lpn + lane) - 1) /
                run.lane_width + 1;
            if (pages_left < lane_pages)
                run.end_lpn = lpn + lane + pages_left * run.lane_width;
            run.lanes.push_back(address);
        }
        ++stats_.compact_read_runs;
    }
    if (run.lanes.empty()) return std::nullopt;
    const auto offset = lpn - run.first_lpn;
    const auto ordinal = offset / run.lane_width;
    auto address = run.lanes[static_cast<std::size_t>(offset % run.lane_width)];
    address.ppn += ordinal;
    address.page += static_cast<std::uint32_t>(ordinal);
    ++stats_.compact_read_run_pages;
    if (ordinal != 0) ++stats_.compact_read_run_reuses;
    return address;
}

std::optional<std::uint64_t> HbfController::compact_lpn_ppn(
    std::uint64_t lpn, const CompactReadAddress* address) const {
    if (!compact_logical_image_ ||
        lpn < compact_logical_image_->first_lpn ||
        lpn - compact_logical_image_->first_lpn >=
            compact_logical_image_->page_count) {
        return std::nullopt;
    }
    if (compact_logical_image_->retired_lpns.contains(lpn)) return std::nullopt;
    const auto resolved = address != nullptr ? *address : compact_lpn_address(lpn);
    const auto& block = blocks_.at(resolved.block_index);
    if (block.role != BlockRole::Data || block.erase_pending ||
        !block.is_valid(resolved.page)) {
        throw std::runtime_error(
            "HBF compact LPN maps to a non-live data page");
    }
    return resolved.ppn;
}

std::optional<std::uint64_t> HbfController::compact_mapping_ppn(
    std::uint64_t mapping_vpn) const {
    if (!compact_logical_image_ ||
        mapping_vpn < compact_logical_image_->first_vpn ||
        mapping_vpn - compact_logical_image_->first_vpn >=
            compact_logical_image_->vpn_slot_count) {
        return std::nullopt;
    }
    if (compact_logical_image_->retired_mapping_vpns.contains(
            mapping_vpn)) {
        return std::nullopt;
    }
    const auto offset = mapping_vpn - compact_logical_image_->first_vpn;
    const auto compact_ppn = compact_logical_image_->mapping_ppns.at(
        static_cast<std::size_t>(offset));
    if (!compact_ppn) {
        return std::nullopt;
    }
    const auto ppn = *compact_ppn;
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.device.pages_per_block);
    const auto page = static_cast<std::uint32_t>(
        ppn % config_.device.pages_per_block);
    const auto& block = blocks_.at(block_index);
    if (block.role != BlockRole::Mapping || block.erase_pending ||
        !block.is_valid(page)) {
        throw std::runtime_error(
            "HBF compact VPN maps to a non-live mapping page");
    }
    return ppn;
}

std::optional<std::uint64_t> HbfController::compact_lpn_for_ppn(
    std::uint64_t ppn) const {
    if (!compact_logical_image_ || ppn >= total_pages_) {
        return std::nullopt;
    }
    const auto& image = *compact_logical_image_;
    const auto block_index = ppn / config_.device.pages_per_block;
    const auto location = image.data_block_locations.find(block_index);
    if (location == image.data_block_locations.end()) {
        return std::nullopt;
    }
    const auto pps = static_cast<std::uint64_t>(planes_per_stack());
    const auto plane = location->second.plane;
    const auto stack = stack_of_plane(plane);
    const auto local_plane =
        static_cast<std::uint64_t>(plane - stack * planes_per_stack());
    const auto page_ordinal = checked_add(
        checked_mul(
            location->second.block_ordinal,
            config_.device.pages_per_block,
            "HBF compact inverse block ordinal"),
        ppn % config_.device.pages_per_block,
        "HBF compact inverse page ordinal");
    const auto data_index = checked_add(
        checked_mul(
            page_ordinal,
            pps,
            "HBF compact inverse striped data index"),
        local_plane,
        "HBF compact inverse data index");
    const auto& offsets = image.vpn_offsets_by_stack.at(stack);
    std::size_t lower = 0;
    std::size_t upper = offsets.size();
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2;
        const auto& range = image.vpn_ranges.at(
            static_cast<std::size_t>(offsets[middle]));
        if (range.stack_page_offset <= data_index) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    if (lower == 0) {
        return std::nullopt;
    }
    const auto vpn_offset = offsets[lower - 1];
    const auto& range = image.vpn_ranges.at(
        static_cast<std::size_t>(vpn_offset));
    if (data_index - range.stack_page_offset >= range.page_count) {
        return std::nullopt;
    }
    const auto mapping_vpn = checked_add(
        image.first_vpn,
        vpn_offset,
        "HBF compact inverse mapping VPN");
    const auto stacks = static_cast<std::uint64_t>(config_.device.stacks);
    const auto group = mapping_vpn / stacks;
    const auto rotation = placement_mix64(group) % stacks;
    const auto lane = static_cast<std::uint64_t>(stack) >= rotation ?
        static_cast<std::uint64_t>(stack) - rotation :
        stacks - (rotation - static_cast<std::uint64_t>(stack));
    const auto local_entry = checked_add(
        range.first_entry,
        data_index - range.stack_page_offset,
        "HBF compact inverse local entry");
    const auto stripe = checked_add(
        checked_mul(
            group,
            config_.host.mapping_entries_per_page,
            "HBF compact inverse mapping stripe"),
        local_entry,
        "HBF compact inverse stripe");
    const auto lpn = checked_add(
        checked_mul(stripe, stacks, "HBF compact inverse LPN stripe"),
        lane,
        "HBF compact inverse LPN");
    if (lpn < image.first_lpn ||
        lpn - image.first_lpn >= image.page_count ||
        image.retired_lpns.contains(lpn)) {
        return std::nullopt;
    }
    const auto forward = compact_lpn_ppn(lpn);
    if (!forward || *forward != ppn) {
        throw std::runtime_error(
            "HBF compact forward/inverse data mapping diverged");
    }
    return lpn;
}

std::optional<HbfController::CompactPageIdentity>
HbfController::compact_page_identity(std::uint64_t ppn) const {
    if (!compact_logical_image_) {
        return std::nullopt;
    }
    const auto& image = *compact_logical_image_;
    if (const auto mapping = image.mapping_vpn_by_ppn.find(ppn);
        mapping != image.mapping_vpn_by_ppn.end() &&
        !image.retired_mapping_vpns.contains(mapping->second)) {
        const auto forward = compact_mapping_ppn(mapping->second);
        if (!forward || *forward != ppn) {
            throw std::runtime_error(
                "HBF compact forward/inverse mapping-page directory "
                "diverged");
        }
        return CompactPageIdentity{
            .logical_key = metadata_lpn(mapping->second),
            .owner = PageOwner::Mapping,
        };
    }
    if (const auto lpn = compact_lpn_for_ppn(ppn)) {
        return CompactPageIdentity{
            .logical_key = *lpn,
            .owner = PageOwner::Logical,
        };
    }
    return std::nullopt;
}

void HbfController::retire_compact_page(
    std::uint64_t logical_key,
    PageOwner owner) {
    if (!compact_logical_image_ ||
        !compact_logical_image_->mutable_image) {
        throw std::runtime_error(
            "HBF cannot retire a non-mutable compact logical page");
    }
    auto& image = *compact_logical_image_;
    std::optional<std::uint64_t> ppn;
    std::unordered_map<std::uint64_t, std::uint32_t>* live_pages = nullptr;
    if (owner == PageOwner::Logical) {
        ppn = compact_lpn_ppn(logical_key);
        if (!ppn || !image.retired_lpns.insert(logical_key)) {
            throw std::runtime_error(
                "HBF compact logical page was already retired");
        }
        record_mutated_lpn_range(logical_key, 1);
        live_pages = &image.live_data_pages_by_block;
    } else if (owner == PageOwner::Mapping &&
               is_metadata_lpn(logical_key)) {
        const auto mapping_vpn = metadata_vpn(logical_key);
        ppn = compact_mapping_ppn(mapping_vpn);
        if (!ppn ||
            !image.retired_mapping_vpns.insert(mapping_vpn)) {
            throw std::runtime_error(
                "HBF compact mapping page was already retired");
        }
        live_pages = &image.live_mapping_pages_by_block;
    } else {
        throw std::runtime_error(
            "HBF compact retirement requires logical or mapping ownership");
    }

    const auto block_index = *ppn / config_.device.pages_per_block;
    auto live = live_pages->find(block_index);
    if (live == live_pages->end() || live->second == 0) {
        throw std::runtime_error(
            "HBF compact retirement lost its live block count");
    }
    if (--live->second == 0) {
        live_pages->erase(live);
    }
    if (programmed_pages_.contains(*ppn)) {
        throw std::runtime_error(
            "HBF compact retirement collided with materialized page state");
    }
    invalidate_live_page(*ppn);
}

std::uint64_t HbfController::logical_mapping_entry_count() const {
    std::uint64_t entries = lpn_to_ppn_.size();
    if (compact_logical_image_) {
        const auto compact_live =
            compact_logical_image_->page_count -
            compact_logical_image_->retired_lpns.size();
        entries = checked_add(
            entries, compact_live, "HBF compact logical mapping entries");
    }
    for (const auto& [lpn, updates] : pending_lpn_updates_) {
        const bool has_live_update = std::any_of(
            updates.begin(), updates.end(), [this](const PendingMappingUpdate& update) {
                const auto block_index = static_cast<std::size_t>(
                    update.new_ppn / config_.device.pages_per_block);
                return blocks_.at(block_index).epoch == update.block_epoch &&
                    !blocks_.at(block_index).erase_pending;
            });
        if (has_live_update &&
            !lpn_to_ppn_.contains(lpn) &&
            !compact_lpn_ppn(lpn)) {
            ++entries;
        }
    }
    return entries;
}

std::size_t HbfController::source_index(TransactionSource source) const {
    switch (source) {
    case TransactionSource::User:
        return 0;
    case TransactionSource::Mapping:
        return 1;
    case TransactionSource::Host:
        return 2;
    case TransactionSource::Prepopulate:
        return 3;
    }
    throw std::runtime_error("unknown HBF transaction source");
}

std::string HbfController::source_name(TransactionSource source) const {
    switch (source) {
    case TransactionSource::User:
        return "user";
    case TransactionSource::Mapping:
        return "mapping";
    case TransactionSource::Host:
        return "gc";
    case TransactionSource::Prepopulate:
        return "prepopulate";
    }
    throw std::runtime_error("unknown HBF transaction source");
}

HeatmapTrafficSource HbfController::resolve_heatmap_source(
    TransactionSource source,
    HeatmapTrafficSource attribution) const {
    const auto attribution_index = static_cast<std::size_t>(attribution);
    if (attribution_index >= kHeatmapTrafficSourceCount) {
        throw std::runtime_error("invalid HBF heatmap traffic source");
    }

    HeatmapTrafficSource expected = attribution;
    switch (source) {
    case TransactionSource::User:
        return attribution;
    case TransactionSource::Mapping:
        expected = HeatmapTrafficSource::Mapping;
        break;
    case TransactionSource::Host:
        if (attribution == HeatmapTrafficSource::Maintenance) return attribution;
        expected = HeatmapTrafficSource::GarbageCollection;
        break;
    case TransactionSource::Prepopulate:
        expected = HeatmapTrafficSource::Prepopulate;
        break;
    default:
        throw std::runtime_error("unknown HBF transaction source");
    }
    if (attribution != expected) {
        throw std::runtime_error(
            "HBF transaction source and heatmap attribution disagree");
    }
    return attribution;
}

std::string HbfController::kind_name(TransactionKind kind) const {
    switch (kind) {
    case TransactionKind::Read:
        return "read";
    case TransactionKind::Program:
        return "program";
    case TransactionKind::Erase:
        return "erase";
    }
    throw std::runtime_error("unknown HBF transaction kind");
}

std::string HbfController::role_name(BlockRole role) const {
    switch (role) {
    case BlockRole::Free:
        return "free";
    case BlockRole::StaticReadOnly:
        return "static-read-only";
    case BlockRole::RawPhysical:
        return "raw-physical";
    case BlockRole::Data:
        return "data";
    case BlockRole::Mapping:
        return "mapping";
    case BlockRole::GC:
        return "gc";
    }
    throw std::runtime_error("unknown HBF block role");
}

void HbfController::flush_all_dirty_mapping_pages(
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    // Bound each batch to one block's page count times the stack count.
    // An unbounded wave pins
    // every old checkpoint until its replacement publication is applied.
    // Under capacity pressure that forces GC to copy data to finance more
    // mapping writes, which in turn dirties still more mapping generations.
    const auto batch_pages = static_cast<std::size_t>(config_.device.stacks) *
        config_.device.pages_per_block;
    std::size_t wave = 0;
    while (!dirty_mapping_vpns_.empty()) {
        std::vector<std::uint64_t> dirty{
            dirty_mapping_vpns_.begin(), dirty_mapping_vpns_.end()};
        std::sort(dirty.begin(), dirty.end());
        for (std::size_t begin = 0; begin < dirty.size(); begin += batch_pages) {
            advance_administrative_frontier(at_ns);
            const double drain_start_ns = at_ns;
            double drain_finish_ns = at_ns;
            const auto end = std::min(dirty.size(), begin + batch_pages);
            for (auto index = begin; index < end; ++index) {
                const auto mapping_vpn = dirty[index];
                const auto events = pending_dirty_mapping_events_.find(mapping_vpn);
                if (events == pending_dirty_mapping_events_.end() ||
                    events->second.empty()) {
                    clear_mapping_page_dirty(mapping_vpn);
                    continue;
                }
                double entry_ns = std::max(
                    drain_start_ns, events->second.rbegin()->first);
                flush_dirty_mapping_page(mapping_vpn, entry_ns, breakdown, spans);
                drain_finish_ns = std::max(drain_finish_ns, entry_ns);
            }
            at_ns = drain_finish_ns;
            // This is a persistence barrier: all foreground work predates
            // it. Retire completed checkpoint copies before admitting the
            // next batch; keep the batch's ordinary media parallelism.
            apply_commits_through(at_ns);
            complete_active_relocations(at_ns, breakdown, spans);
            if (!pending_commits_.empty())
                at_ns = std::max(at_ns, pending_commits_.rbegin()->first.first);
            apply_commits_through(at_ns);
        }
        if (++wave > blocks_.size() + 1) {
            throw std::runtime_error(
                "HBF mapping drain failed to converge after GC-generated updates");
        }
    }
}

void HbfController::mark_mapping_page_dirty(
    std::uint64_t mapping_vpn,
    double at_ns) {
    const auto sequence = next_cache_touch_sequence_++;
    auto& events = pending_dirty_mapping_events_[mapping_vpn];
    events.emplace(at_ns, sequence);
    // A checkpoint snapshots the newest event at or before its own time,
    // and every later query time is at or after the causal watermark, so
    // only the newest event at or behind the watermark and the events
    // beyond it can ever be selected. Older ones are unobservable; without
    // this, full-resident mapping kept one node per page write between
    // drains.
    const auto first_future = events.upper_bound(TemporalTouch{
        reservation_causal_watermark_ns_,
        std::numeric_limits<std::uint64_t>::max()});
    if (first_future != events.begin()) {
        events.erase(events.begin(), std::prev(first_future));
    }
    set_mapping_page_dirty(mapping_vpn);
}

void HbfController::set_mapping_page_dirty(std::uint64_t mapping_vpn) {
    if (dirty_mapping_vpns_.insert(mapping_vpn).second) {
        dirty_mapping_pages_by_stack_.at(stack_for_vpn(mapping_vpn))++;
    }
}

void HbfController::clear_mapping_page_dirty(std::uint64_t mapping_vpn) {
    if (dirty_mapping_vpns_.erase(mapping_vpn) != 0) {
        auto& count = dirty_mapping_pages_by_stack_.at(stack_for_vpn(mapping_vpn));
        if (count == 0) {
            throw std::runtime_error("HBF dirty mapping-page count underflowed");
        }
        count--;
    }
}

void HbfController::touch_mapping_cache_entry(
    std::size_t stack,
    std::uint64_t mapping_vpn) {
    auto& cache = mapping_cache_by_stack_.at(stack);
    auto found = cache.find(mapping_vpn);
    if (found == cache.end()) {
        throw std::runtime_error(
            "HBF mapping-cache touch names a nonresident page");
    }
    auto& lru = mapping_cache_lru_by_stack_.at(stack);
    // splice preserves the node/iterator and is also a no-op for the MRU.
    lru.splice(lru.begin(), lru, found->second.iterator);
}

std::uint64_t HbfController::mapping_cache_key(std::uint64_t lpn) const {
    return config_.host.mapping_cache_layout == MappingCacheLayout::Entry ? lpn : mapping_vpn_for_lpn(lpn);
}

std::uint64_t HbfController::first_lpn_for_mapping_vpn(std::uint64_t vpn) const {
    const auto stacks = static_cast<std::uint64_t>(config_.device.stacks);
    const auto group = vpn / stacks;
    const auto rotation = placement_mix64(group) % stacks;
    const auto lane = (vpn % stacks + stacks - rotation) % stacks;
    return group * config_.host.mapping_entries_per_page * stacks + lane;
}

std::uint64_t HbfController::mapping_cache_record_bytes(
    std::uint64_t mapping_vpn, double at_ns, bool for_update) const {
    const auto tag = config_.host.mapping_cache_tag_bytes;
    if (config_.host.mapping_cache_layout == MappingCacheLayout::Entry) return tag + 8;
    if (config_.host.mapping_cache_layout == MappingCacheLayout::Page || for_update)
        return tag + config_.device.page_size_bytes;
    // Each 24-byte affine descriptor stores index/length, PPN base and signed
    // stride. Unmapped entries form separate sentinel runs. Inspect only the
    // mapping page whose real media read has just completed.
    const auto first_lpn = first_lpn_for_mapping_vpn(mapping_vpn);
    std::uint64_t runs = 0;
    std::uint64_t length = 0;
    std::optional<std::uint64_t> previous;
    std::int64_t stride = 0;
    for (std::uint64_t i = 0; i < config_.host.mapping_entries_per_page; ++i) {
        const auto lpn = first_lpn + i * config_.device.stacks;
        const auto current = lpn < total_pages_ ? visible_lpn_at(lpn, at_ns) : std::nullopt;
        bool extend = length != 0 && current.has_value() == previous.has_value();
        std::int64_t delta = 0;
        if (extend && current) {
            delta = static_cast<std::int64_t>(*current) - static_cast<std::int64_t>(*previous);
            extend = length == 1 || delta == stride;
        }
        if (!extend) { ++runs; length = 1; }
        else { if (length == 1) stride = delta; ++length; }
        previous = current;
    }
    return tag + std::min(config_.device.page_size_bytes, runs * 24);
}

void HbfController::attach_hbm_buffer(physical::hbm::HbmDevice& hbm) {
    if (buffer_hbm_ || stats_.host_hbm_read_bytes || stats_.host_hbm_write_bytes)
        throw std::runtime_error("HBF HBM buffer must be attached once before traffic");
    if (!mapping_is_local() && config_.host.ctrl_dram_bytes)
        hbm.reserve_controller_buffer(config_.host.ctrl_dram_bytes);
    buffer_hbm_ = &hbm;
    buffer_hbm_cursor_by_stack_.assign(config_.device.stacks, 0);
    stats_.host_hbm_reserved_bytes = mapping_is_local() ? 0 : hbm.controller_buffer_bytes();
}

bool HbfController::mapping_is_local() const {
    return config_.host.mapping_placement == MappingPlacement::DeviceLocal;
}

std::string HbfController::mapping_entity(std::size_t stack) const {
    return mapping_is_local() ? "stack" + std::to_string(stack) + "/mapping" :
        "host/mapping/partition" + std::to_string(stack);
}

void HbfController::service_mapping_compute(std::size_t stack, double work_ns,
    double& at_ns, Breakdown& breakdown, std::vector<TraceSpan>* spans, const char* name) {
    if (work_ns == 0) return;
    const auto per_stack = mapping_compute_workers_.size() / config_.device.stacks;
    const bool partitioned = mapping_is_local() || !config_.host.mapping_host_compute_shared;
    const auto begin = partitioned ? stack * per_stack : 0;
    const auto end = partitioned ? begin + per_stack : mapping_compute_workers_.size();
    auto selected = begin;
    double earliest = std::numeric_limits<double>::infinity();
    double selected_idle_begin = -1.0;
    for (auto i = begin; i < end; ++i) {
        auto& worker = mapping_compute_workers_.at(i);
        worker.prune_before(reservation_causal_watermark_ns_);
        const auto start = worker.preview_start(at_ns, work_ns);
        const auto gap = worker.first_fitting_gap(at_ns, work_ns);
        const auto idle_begin = gap ? gap->begin_ns : worker.ready_ns;
        // A continuation ready at the same instant on several workers should
        // use the most recently freed one. Always picking the first worker
        // can split another worker's long idle interval with a tiny future
        // callback, stranding space that a complete lookup could have used.
        if (start < earliest || (start == earliest && idle_begin > selected_idle_begin)) {
            earliest = start; selected = i; selected_idle_begin = idle_begin;
        }
    }
    const auto work = reserve(at_ns, work_ns, mapping_compute_workers_.at(selected));
    stats_.mapping_compute_work_ns += work_ns;
    stats_.mapping_compute_queue_ns += work.wait_ns;
    ++stats_.mapping_compute_ops;
    breakdown.translation_ns += work_ns;
    breakdown.scheduler_queue_wait_ns += work.wait_ns;
    if (spans) {
        const auto entity = mapping_is_local() ?
            "stack" + std::to_string(stack) + "/mapping/worker" + std::to_string(selected - begin) :
            "host/mapping/worker" + std::to_string(selected);
        trace_wait(spans, entity, at_ns, work.start_ns, "wait_mapping_compute");
        add_trace_span(spans, name, "mapping_compute", entity, work.start_ns, work.finish_ns, true);
    }
    at_ns = work.finish_ns;
}

void HbfController::service_mapping_sram(std::size_t stack, bool write, double& at_ns,
    Breakdown& breakdown, std::vector<TraceSpan>* spans, const char* name, bool foreground) {
    auto& port = mapping_sram_issue_by_stack_.at(stack);
    port.prune_before(reservation_causal_watermark_ns_);
    const auto issue = reserve(at_ns, config_.host.mapping_sram_issue_ns, port);
    const auto finish = std::max(issue.finish_ns, issue.start_ns + config_.host.mapping_sram_latency_ns);
    stats_.mapping_sram_issue_busy_ns += config_.host.mapping_sram_issue_ns;
    stats_.mapping_sram_queue_ns += issue.wait_ns;
    if (write) stats_.mapping_sram_write_bytes += mapping_sram_record_bytes;
    else stats_.mapping_sram_read_bytes += mapping_sram_record_bytes;
    if (foreground) {
        breakdown.translation_ns += finish - issue.start_ns;
        breakdown.scheduler_queue_wait_ns += issue.wait_ns;
    } else {
        breakdown.maintenance_ns += finish - issue.start_ns;
    }
    if (spans) {
        const auto entity = "host/mapping/sram" + std::to_string(stack);
        trace_wait(spans, entity, at_ns, issue.start_ns, "wait_mapping_sram_port");
        add_trace_span(spans, name, "mapping_sram", entity, issue.start_ns, finish,
            foreground, "32B including tag and replacement state");
    }
    at_ns = finish;
}

bool HbfController::probe_mapping_sram(std::uint64_t lpn, MappingAccessKind kind,
    double& at_ns, Breakdown& breakdown, std::vector<TraceSpan>* spans) {
    if (!config_.host.mapping_sram_bytes) return false;
    const auto stack = stack_for_lpn(lpn);
    auto& cache = mapping_sram_by_stack_.at(stack);
    auto& lru = mapping_sram_lru_by_stack_.at(stack);
    service_mapping_sram(stack, false, at_ns, breakdown, spans, "mapping_sram_probe");
    const auto wait_until = [&](double ready) {
        if (ready <= at_ns) return;
        const auto wait = ready - at_ns;
        stats_.mapping_sram_queue_ns += wait;
        breakdown.scheduler_queue_wait_ns += wait;
        if (spans) trace_wait(spans, "host/mapping/sram" + std::to_string(stack),
            at_ns, ready, "wait_mapping_sram_record");
        at_ns = ready;
    };
    const auto found = cache.find(lpn);
    if (kind == MappingAccessKind::Update) {
        // Every update, including trim and GC relocation, still writes HBM.
        // Retire an outstanding old fill before invalidating its physical slot;
        // otherwise it could later overwrite a newly cached mapping.
        if (found != cache.end()) {
            wait_until(found->second.evictable_ns);
            service_mapping_sram(stack, true, at_ns, breakdown, spans, "mapping_sram_invalidate");
            lru.erase(found->second.iterator);
            cache.erase(found);
            --stats_.mapping_sram_entries;
            ++stats_.mapping_sram_invalidations;
        }
        return false;
    }
    if (found != cache.end()) {
        if (found->second.fill_ready_ns <= at_ns) ++stats_.mapping_sram_hits;
        else {
            ++stats_.mapping_sram_misses;
            ++stats_.mapping_sram_coalesced_misses;
            wait_until(found->second.fill_ready_ns);
        }
        found->second.evictable_ns = std::max(found->second.evictable_ns, at_ns);
        lru.splice(lru.begin(), lru, found->second.iterator);
        return true;
    }
    ++stats_.mapping_sram_misses;
    const auto capacity = config_.host.mapping_sram_bytes /
        config_.device.stacks / mapping_sram_record_bytes;
    if (cache.size() == capacity) {
        const auto victim = cache.find(lru.back());
        wait_until(victim->second.evictable_ns);
        cache.erase(victim);
        lru.pop_back();
        --stats_.mapping_sram_entries;
        ++stats_.mapping_sram_evictions;
    }
    return false;
}

void HbfController::fill_mapping_sram(std::uint64_t lpn, double at_ns,
    Breakdown& breakdown, std::vector<TraceSpan>* spans) {
    if (!config_.host.mapping_sram_bytes) return;
    const auto stack = stack_for_lpn(lpn);
    auto& cache = mapping_sram_by_stack_.at(stack);
    auto& lru = mapping_sram_lru_by_stack_.at(stack);
    // The triggering request can use the HBM response immediately. The fill
    // occupies the SRAM port in parallel; future hits/evictions wait for it.
    service_mapping_sram(stack, true, at_ns, breakdown, spans, "mapping_sram_fill", false);
    lru.push_front(lpn);
    const auto inserted = cache.emplace(lpn, MappingSramEntry{
        .fill_ready_ns = at_ns, .evictable_ns = at_ns, .iterator = lru.begin()});
    if (!inserted.second) throw std::runtime_error("duplicate mapping SRAM fill");
    ++stats_.mapping_sram_fills;
    ++stats_.mapping_sram_entries;
    stats_.mapping_sram_peak_entries = std::max(stats_.mapping_sram_peak_entries, stats_.mapping_sram_entries);
}

double HbfController::mapping_memory_transfer(std::size_t stack, std::uint64_t bytes,
    Op op, double at_ns, Breakdown& breakdown, std::vector<TraceSpan>* spans) {
    if (!mapping_is_local()) return host_memory_transfer(stack, bytes, op, at_ns, breakdown, spans);
    if (!bytes) return at_ns;
    const auto burst = config_.host.local_mapping_burst_bytes;
    const auto rounded = (bytes + burst - 1) / burst * burst;
    const auto duration = transfer_time_ns(rounded, config_.host.local_mapping_bandwidth_GBps);
    const auto transfer = reserve(at_ns, duration, local_mapping_bus_by_stack_.at(stack));
    if (op == Op::Read) stats_.local_mapping_read_bytes += rounded;
    else stats_.local_mapping_write_bytes += rounded;
    stats_.local_mapping_busy_ns += duration;
    stats_.local_mapping_queue_ns += transfer.wait_ns;
    breakdown.maintenance_ns += duration;
    breakdown.scheduler_queue_wait_ns += transfer.wait_ns;
    if (spans) {
        trace_wait(spans, mapping_entity(stack), at_ns, transfer.start_ns, "wait_local_mapping_memory");
        add_trace_span(spans, op == Op::Read ? "local_mapping_read" : "local_mapping_write",
            "mapping_memory", mapping_entity(stack), transfer.start_ns, transfer.finish_ns, true,
            std::to_string(rounded) + "B");
    }
    return transfer.finish_ns;
}

double HbfController::host_memory_transfer(std::size_t stack, std::uint64_t bytes,
    Op op, double earliest_ns, Breakdown& breakdown, std::vector<TraceSpan>* spans) {
    if (!bytes) return earliest_ns;
    if (!buffer_hbm_)
        throw std::runtime_error("HBF controller storage requires an explicitly attached HBM device");
    const auto burst = buffer_hbm_->config().burst_bytes();
    const auto partition = config_.host.ctrl_dram_bytes / config_.device.stacks / burst * burst;
    if (!partition || bytes > partition || stack >= config_.device.stacks)
        throw std::runtime_error("HBF buffer transfer exceeds its reserved HBM partition");
    auto& cursor = buffer_hbm_cursor_by_stack_.at(stack);
    const auto rounded = (bytes + burst - 1) / burst * burst;
    if (cursor > partition - rounded) cursor = 0;
    // Channel-level traffic striping within each capacity-bounded partition.
    // These addresses do not claim row locality or a physical cache allocator.
    const auto before = buffer_hbm_->execution_stats().controller_buffer_bus_busy_ns;
    const auto completion = buffer_hbm_->transfer_controller_buffer(PhysicalRequest{
        .id = "hbf-controller-buffer", .tier = Tier::HBM, .op = op,
        .address_space = AddressSpace::Physical,
        .trace = TraceConfig{.mode = spans ? physical::TraceMode::Full : physical::TraceMode::Off},
        .arrival_ns = earliest_ns,
        .addr = buffer_hbm_->application_capacity_bytes() + stack * partition + cursor,
        .bytes = bytes,
    });
    cursor += rounded;
    if (op == Op::Read) stats_.host_hbm_read_bytes += completion.physical_bytes;
    else stats_.host_hbm_write_bytes += completion.physical_bytes;
    stats_.host_hbm_busy_ns += buffer_hbm_->execution_stats().controller_buffer_bus_busy_ns - before;
    stats_.host_hbm_queue_wait_ns += completion.breakdown.scheduler_queue_wait_ns;
    breakdown.maintenance_ns += completion.breakdown.channel_transfer_ns;
    breakdown.scheduler_queue_wait_ns += completion.breakdown.scheduler_queue_wait_ns;
    if (spans) spans->insert(spans->end(), completion.spans.begin(), completion.spans.end());
    return completion.finish_ns;
}

void HbfController::mapping_dram_access(
    double& at_ns, std::size_t stack, Breakdown& breakdown,
    std::vector<TraceSpan>* spans, const char* name, std::uint64_t& counter, bool write) {
    service_mapping_compute(stack, config_.host.mapping_internal_compute_ns,
        at_ns, breakdown, spans, "mapping_bookkeeping");
    const auto issue = reserve(at_ns, config_.host.ctrl_dram_issue_ns,
        mapping_memory_issue_by_stack_.at(stack));
    stats_.mapping_dram_issue_busy_ns += config_.host.ctrl_dram_issue_ns;
    ++counter;
    if (issue.wait_ns > 0.0) {
        ++stats_.mapping_dram_wait_ops;
        stats_.mapping_dram_wait_ns += issue.wait_ns;
        stats_.mapping_dram_wait_max_ns = std::max(stats_.mapping_dram_wait_max_ns, issue.wait_ns);
        breakdown.scheduler_queue_wait_ns += issue.wait_ns;
        if (spans) trace_wait(spans, mapping_entity(stack), at_ns, issue.start_ns, name);
    }
    const auto data_done = mapping_memory_transfer(stack, config_.host.mapping_directory_entry_bytes,
        write ? Op::Write : Op::Read, issue.start_ns, breakdown, spans);
    at_ns = std::max({issue.finish_ns, data_done, issue.start_ns + config_.host.ctrl_dram_latency_ns});
    breakdown.mapping_dram_ns += config_.host.ctrl_dram_latency_ns;
    if (spans) add_trace_span(spans, name, "metadata", mapping_entity(stack),
        issue.start_ns, at_ns, true, "mapping metadata");
}

HbfController::MappingScratchPage* HbfController::mapping_buffer_page(
    std::size_t stack, std::uint64_t vpn) {
    auto& pages = mapping_scratch_by_stack_.at(stack);
    const auto found = std::find_if(pages.begin(), pages.end(),
        [vpn](const auto& page) { return page.vpn == vpn; });
    return found == pages.end() ? nullptr : &*found;
}

HbfController::MappingScratchPage* HbfController::acquire_mapping_scratch(
    std::size_t stack, double& at_ns, Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    auto& pages = mapping_scratch_by_stack_.at(stack);
    if (pages.empty()) return nullptr;
    auto slot = std::min_element(pages.begin(), pages.end(), [&](const auto& a, const auto& b) {
        if (config_.host.mapping_cache_layout != MappingCacheLayout::Entry)
            return a.evictable_ns < b.evictable_ns;
        if (a.vpn.has_value() != b.vpn.has_value()) return !a.vpn;
        return a.touch < b.touch;
    });
    if (slot->evictable_ns > at_ns) {
        breakdown.scheduler_queue_wait_ns += slot->evictable_ns - at_ns;
        if (spans) trace_wait(spans, logic_entity(stack), at_ns,
            slot->evictable_ns, "wait_mapping_scratch");
        at_ns = slot->evictable_ns;
    }
    slot->vpn.reset();
    slot->touch = next_cache_touch_sequence_++;
    return &*slot;
}

bool HbfController::mapping_entry_dirty(const MappingCacheEntry& entry) const {
    return dirty_mapping_vpns_.contains(entry.backing_vpn) &&
        (config_.host.mapping_cache_layout != MappingCacheLayout::Entry ||
         entry.modified_ns > mapping_checkpointed_through_ns_.at(entry.backing_vpn));
}

void HbfController::merge_cached_mapping_entries(
    std::uint64_t vpn, double& at_ns, Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    if (!dirty_mapping_vpns_.contains(vpn)) return;
    const auto stack = stack_for_vpn(vpn);
    auto& cache = mapping_cache_by_stack_.at(stack);
    const auto first_lpn = first_lpn_for_mapping_vpn(vpn);
    // No unbudgeted reverse index: probe the bounded set of LPN tags covered
    // by this translation page. The tag reads and dirty overlays are charged
    // to the same pipelined DRAM port as ordinary translation accesses.
    double ready = at_ns;
    for (std::uint64_t i = 0; i < config_.host.mapping_entries_per_page; ++i) {
        const auto found = cache.find(first_lpn + i * config_.device.stacks);
        if (found != cache.end() && mapping_entry_dirty(found->second))
            ready = std::max(ready, found->second.evictable_ns);
    }
    breakdown.scheduler_queue_wait_ns += ready - at_ns;
    const double probe_start = ready;
    for (std::uint64_t i = 0; i < config_.host.mapping_entries_per_page; ++i) {
        const auto lpn = first_lpn + i * config_.device.stacks;
        double probe_done = probe_start;
        mapping_dram_access(probe_done, stack, breakdown, spans,
            "mapping_dirty_entry_probe", stats_.mapping_dirty_probe_ops);
        ready = std::max(ready, probe_done);
        const auto found = cache.find(lpn);
        if (found != cache.end() && mapping_entry_dirty(found->second)) {
            found->second.evictable_ns = std::max(found->second.evictable_ns, probe_done);
            // Stream each dirty value into the reserved page as its probe
            // completes; no additional reverse index or staging list.
            mapping_dram_access(probe_done, stack, breakdown, spans,
                "mapping_buffer_overlay", stats_.mapping_buffer_patch_ops, true);
            ready = std::max(ready, probe_done);
        }
    }
    at_ns = ready;
}

void HbfController::service_mapping_codec(
    std::uint64_t vpn, double& at_ns, Breakdown& breakdown,
    std::vector<TraceSpan>* spans, const char* name, MappingScratchPage* workspace) {
    const auto stack = stack_for_vpn(vpn);
    // Encoding keeps the fill buffer through compute completion. Only an
    // expansion of an already cached compressed record needs a new workspace.
    auto* slot = workspace ? workspace : acquire_mapping_scratch(stack, at_ns, breakdown, spans);
    const auto work = config_.host.mapping_entries_per_page * config_.host.mapping_codec_ns_per_entry;
    if (work > 0.0) {
        stats_.mapping_codec_work_ns += work;
        service_mapping_compute(stack, work, at_ns, breakdown, spans, name);
    }
    if (slot) slot->fill_ready_ns = slot->evictable_ns = at_ns;
}

HbfController::MappingScratchPage* HbfController::read_mapping_backing_page(
    std::uint64_t mapping_vpn, double& at_ns, Breakdown& breakdown,
    std::vector<TraceSpan>* spans, bool merge) {
    const auto stack = stack_for_vpn(mapping_vpn);
    if (config_.host.mapping_cache_layout == MappingCacheLayout::Entry) {
        if (auto* page = mapping_buffer_page(stack, mapping_vpn)) {
            const bool pending = page->fill_ready_ns > at_ns;
            const auto ready = merge ? page->evictable_ns : page->fill_ready_ns;
            if (ready > at_ns) {
                breakdown.scheduler_queue_wait_ns += ready - at_ns;
                if (spans) trace_wait(spans, logic_entity(stack), at_ns, ready, "wait_mapping_buffer_fill");
                at_ns = ready;
            }
            mapping_dram_access(at_ns, stack, breakdown, spans,
                "mapping_buffer_read", stats_.mapping_buffer_lookup_ops);
            page->evictable_ns = std::max(page->evictable_ns, at_ns);
            page->touch = next_cache_touch_sequence_++;
            if (merge) ++stats_.mapping_merge_buffer_hits;
            else {
                ++stats_.mapping_cache_buffer_hits;
                if (pending) ++stats_.mapping_buffer_coalesced_reads;
            }
            return page;
        }
        mapping_dram_access(at_ns, stack, breakdown, spans,
            "mapping_buffer_probe", stats_.mapping_buffer_lookup_ops);
    }
    mapping_dram_access(at_ns, stack, breakdown, spans,
        "mapping_directory_lookup", stats_.mapping_directory_lookup_ops);
    auto* scratch = acquire_mapping_scratch(stack, at_ns, breakdown, spans);
    wait_for_pending_vpn_erase(
        mapping_vpn, at_ns, breakdown, spans);
    auto backing_ppn = visible_mapping_vpn_at(mapping_vpn, at_ns);
    if (backing_ppn) {
        wait_for_pending_block_erase(
            *backing_ppn, at_ns, breakdown, spans);
        if (pending_block_transitions_.contains(static_cast<std::size_t>(
                *backing_ppn / config_.device.pages_per_block))) {
            // The persisted copy sits in a block whose GC reclaim is already
            // issued, so its relocation is published: wait for that
            // publication and read the relocated copy.
            double relocation_ready_ns = at_ns;
            std::unordered_set<std::uint64_t> sequences;
            if (const auto pending = pending_vpn_updates_.find(mapping_vpn);
                pending != pending_vpn_updates_.end()) {
                for (const auto& update : pending->second) {
                    relocation_ready_ns = std::max(
                        relocation_ready_ns, update.commit_ns);
                    sequences.insert(update.program_commit_sequence);
                    sequences.insert(update.sequence);
                }
            }
            if (relocation_ready_ns > at_ns) {
                breakdown.scheduler_queue_wait_ns += relocation_ready_ns - at_ns;
                if (spans != nullptr) {
                    trace_wait(
                        spans,
                        logic_entity(static_cast<std::uint32_t>(stack)),
                        at_ns,
                        relocation_ready_ns,
                        "wait_mapping_page_relocation");
                }
                at_ns = relocation_ready_ns;
            }
            apply_selected_commits_through(sequences, at_ns);
            backing_ppn = visible_mapping_vpn_at(mapping_vpn, at_ns);
            if (!backing_ppn) {
                throw std::runtime_error(
                    "HBF persisted mapping page vanished under a GC reclaim");
            }
        }
        at_ns = schedule_read_page(
            *backing_ppn,
            at_ns,
            breakdown,
            spans,
            TransactionSource::Mapping,
            HeatmapTrafficSource::Mapping,
            ReadPayloadRoute::MappingBuffer,
            0);
        stats_.physical_read_bytes = checked_add(
            stats_.physical_read_bytes,
            config_.device.page_size_bytes,
            "HBF mapping-cache physical read bytes");
        stats_.page_reads = checked_add(
            stats_.page_reads, 1, "HBF mapping-cache page reads");
        if (merge) {
            stats_.mapping_merge_media_reads++;
        } else {
            stats_.mapping_media_reads++;
            stats_.mapping_media_read_bytes += config_.device.page_size_bytes;
        }
    } else {
        // A small stack-local directory can identify mapping groups that have
        // never been persisted. Their miss installs an erased page image
        // without fabricating a NAND read.
        if (!merge) stats_.mapping_cache_erased_misses++;
        at_ns = mapping_memory_transfer(stack, config_.device.page_size_bytes,
            Op::Write, at_ns, breakdown, spans);
    }

    if (scratch) {
        if (config_.host.mapping_cache_layout == MappingCacheLayout::Entry) {
            merge_cached_mapping_entries(mapping_vpn, at_ns, breakdown, spans);
            scratch->vpn = mapping_vpn;
        }
        scratch->fill_ready_ns = scratch->evictable_ns = at_ns;
    }
    return scratch;
}

void HbfController::evict_mapping_cache_record(
    std::size_t stack, double& at_ns, Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    auto& cache = mapping_cache_by_stack_.at(stack);
    auto& lru = mapping_cache_lru_by_stack_.at(stack);
    const auto candidate = std::find_if(lru.rbegin(), lru.rend(),
        [&](auto key) { return cache.at(key).pins == 0; });
    if (candidate == lru.rend())
        throw std::runtime_error("mapping cache has no unpinned eviction candidate");
    const auto key = *candidate;
    auto victim = cache.find(key);
    if (victim->second.evictable_ns > at_ns) {
        breakdown.scheduler_queue_wait_ns += victim->second.evictable_ns - at_ns;
        if (spans) trace_wait(spans, logic_entity(stack), at_ns,
            victim->second.evictable_ns, "wait_mapping_cache_victim");
        at_ns = victim->second.evictable_ns;
    }
    const auto vpn = victim->second.backing_vpn;
    const bool dirty = mapping_entry_dirty(victim->second);
    // A dirty entry is the only resident copy of its update until merged.
    // Keep it pinned while allocating/merging the checkpoint, including any
    // intervening GC. Sibling entries are cleaned by the persisted epoch.
    if (dirty && config_.host.mapping_cache_layout == MappingCacheLayout::Entry) {
        ++victim->second.pins;
        do {
            flush_dirty_mapping_page(vpn, at_ns, breakdown, spans);
        } while (mapping_entry_dirty(cache.at(key)));
        --cache.at(key).pins;
        victim = cache.find(key);
    }
    mapping_cache_bytes_by_stack_.at(stack) -= victim->second.charged_bytes;
    stats_.mapping_cache_live_bytes -= victim->second.charged_bytes;
    lru.erase(victim->second.iterator);
    cache.erase(victim);
    --stats_.mapping_cache_entries;
    ++stats_.mapping_cache_evictions;
    if (dirty) {
        ++stats_.mapping_cache_dirty_evictions;
        if (config_.host.mapping_cache_layout != MappingCacheLayout::Entry)
            flush_dirty_mapping_page(vpn, at_ns, breakdown, spans);
    }
}

void HbfController::ensure_mapping_page_cached(
    std::uint64_t mapping_vpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    std::optional<std::uint64_t> requested_lpn,
    bool for_update) {
    if (config_.host.mapping_mode != MappingMode::Cached) {
        throw std::runtime_error(
            "HBF mapping-cache access used outside cached mode");
    }
    const auto stack = stack_for_vpn(mapping_vpn);
    const auto cache_key = config_.host.mapping_cache_layout == MappingCacheLayout::Entry ?
        requested_lpn.value() : mapping_vpn;
    auto& cache = mapping_cache_by_stack_.at(stack);
    auto& live_bytes = mapping_cache_bytes_by_stack_.at(stack);
    auto& lru = mapping_cache_lru_by_stack_.at(stack);
    // The resident compressed record already contains the complete mapping.
    // Expand it in DRAM, retaining its ownership through capacity eviction
    // and any reentrant GC. No redundant flash read is needed.
    if (for_update && config_.host.mapping_cache_layout == MappingCacheLayout::Extent &&
        cache.contains(cache_key) && !cache.at(cache_key).dense) {
        ++cache.at(cache_key).pins;
        const auto dense_bytes = config_.device.page_size_bytes + config_.host.mapping_cache_tag_bytes;
        while (live_bytes + dense_bytes - cache.at(cache_key).charged_bytes >
               stats_.mapping_cache_capacity_bytes_per_stack)
            evict_mapping_cache_record(stack, at_ns, breakdown, spans);
        auto& entry = cache.at(cache_key);
        if (!entry.dense) {
            if (entry.evictable_ns > at_ns) {
                breakdown.scheduler_queue_wait_ns += entry.evictable_ns - at_ns;
                if (spans) trace_wait(spans, logic_entity(stack), at_ns,
                    entry.evictable_ns, "wait_mapping_cache_promotion");
                at_ns = entry.evictable_ns;
            }
            const auto growth = dense_bytes - entry.charged_bytes;
            live_bytes += growth;
            stats_.mapping_cache_live_bytes += growth;
            stats_.mapping_cache_peak_bytes = std::max(stats_.mapping_cache_peak_bytes,
                stats_.mapping_cache_live_bytes);
            entry.charged_bytes = dense_bytes;
            entry.dense = true;
            service_mapping_codec(mapping_vpn, at_ns, breakdown, spans, "mapping_decode", nullptr);
            entry.fill_ready_ns = entry.evictable_ns = at_ns;
            ++stats_.mapping_cache_promotions;
        }
        --entry.pins;
    }
    const auto finish_existing = [&](bool count_hit) {
        auto found = cache.find(cache_key);
        if (found == cache.end()) {
            throw std::runtime_error(
                "HBF mapping-cache coalesced fill disappeared");
        }
        if (count_hit) {
            stats_.mapping_cache_hits++;
        } else {
            stats_.mapping_cache_coalesced_misses++;
        }
        if (found->second.fill_ready_ns > at_ns) {
            breakdown.scheduler_queue_wait_ns +=
                found->second.fill_ready_ns - at_ns;
            if (spans != nullptr) {
                trace_wait(
                    spans,
                    logic_entity(static_cast<std::uint32_t>(stack)),
                    at_ns,
                    found->second.fill_ready_ns,
                    count_hit ?
                        "wait_mapping_cache_fill" :
                        "wait_mapping_cache_reentrant_fill");
            }
            at_ns = found->second.fill_ready_ns;
        }
        touch_mapping_cache_entry(stack, cache_key);
    };
    if (cache.contains(cache_key)) {
        finish_existing(true);
        return;
    }

    stats_.mapping_cache_misses++;
    // Reserve the worst-case dense fill first. E can release space only
    // after the on-media page has arrived and its actual runs are encoded.
    const auto required_bytes = config_.host.mapping_cache_tag_bytes +
        (config_.host.mapping_cache_layout == MappingCacheLayout::Entry ? 8 : config_.device.page_size_bytes);
    while (live_bytes + required_bytes > stats_.mapping_cache_capacity_bytes_per_stack) {
        // A dirty-victim checkpoint can invoke GC, whose mapping update may
        // recursively fill the same VPN this outer miss was resolving. That
        // nested access owns the one media fill. The outer access remains a
        // miss for access accounting, but must consume the resident result
        // instead of evicting around it and attempting a duplicate insert.
        if (cache.contains(cache_key)) {
            finish_existing(false);
            return;
        }
        evict_mapping_cache_record(stack, at_ns, breakdown, spans);
        // GC triggered by a dirty eviction may itself access mapping pages.
        // Recheck capacity rather than assuming that one removal left a slot.
    }

    if (cache.contains(cache_key)) {
        finish_existing(false);
        return;
    }

    auto* fill_workspace = read_mapping_backing_page(mapping_vpn, at_ns, breakdown, spans, false);

    if (config_.host.mapping_cache_layout == MappingCacheLayout::Extent && !for_update)
        service_mapping_codec(mapping_vpn, at_ns, breakdown, spans, "mapping_encode", fill_workspace);

    auto charged_bytes = mapping_cache_record_bytes(mapping_vpn, at_ns, for_update);
    const bool dense = charged_bytes == config_.device.page_size_bytes + config_.host.mapping_cache_tag_bytes;
    if (config_.host.mapping_cache_layout == MappingCacheLayout::Extent && !for_update) {
        stats_.mapping_compression_input_bytes += config_.device.page_size_bytes;
        stats_.mapping_compression_output_bytes += charged_bytes - config_.host.mapping_cache_tag_bytes;
    }

    lru.push_front(cache_key);
    const auto inserted = cache.emplace(
        cache_key,
        MappingCacheEntry{
            .fill_ready_ns = at_ns,
            .evictable_ns = at_ns,
            .backing_vpn = mapping_vpn,
            .charged_bytes = charged_bytes,
            .dense = dense,
            .modified_ns = -1.0,
            .iterator = lru.begin(),
        });
    if (!inserted.second) {
        lru.pop_front();
        throw std::runtime_error(
            "HBF mapping-cache miss raced with an existing page");
    }
    live_bytes += charged_bytes;
    stats_.mapping_cache_live_bytes += charged_bytes;
    stats_.mapping_cache_peak_bytes = std::max(stats_.mapping_cache_peak_bytes, stats_.mapping_cache_live_bytes);
    stats_.mapping_cache_entries++;
    stats_.mapping_cache_peak_entries = std::max(
        stats_.mapping_cache_peak_entries,
        stats_.mapping_cache_entries);
}

void HbfController::access_mapping(
    std::uint64_t lpn,
    TransactionSource source,
    MappingAccessKind kind,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const CompactReadAddress* address) {
    if (source != TransactionSource::User &&
        source != TransactionSource::Host) {
        throw std::runtime_error(
            "HBF mapping access source must be foreground user or GC");
    }
    const auto stack = static_cast<std::uint32_t>(
        address != nullptr ? address->stack : stack_for_lpn(lpn));

    if (mapping_is_local() && source == TransactionSource::User && kind == MappingAccessKind::Lookup) {
        // The counterfactual accepts an LPN at a stack-level command ingress.
        // It cannot choose a NAND channel from a PPN before doing translation.
        at_ns = schedule_external_request_command(stack * config_.device.channels_per_stack,
            at_ns, breakdown, spans, "logical FTL request", "user");
    }
    service_mapping_compute(stack, config_.host.mapping_control_compute_ns,
        at_ns, breakdown, spans, kind == MappingAccessKind::Lookup ?
            "mapping_lookup_compute" : "mapping_update_compute");

    if (kind == MappingAccessKind::Lookup) {
        ++stats_.mapping_lookup_ops;
        if (source == TransactionSource::User) ++stats_.mapping_user_lookup_ops;
        else ++stats_.mapping_gc_lookup_ops;
    } else {
        ++stats_.mapping_update_ops;
        if (source == TransactionSource::User) ++stats_.mapping_user_update_ops;
        else ++stats_.mapping_gc_update_ops;
    }
    if (probe_mapping_sram(lpn, kind, at_ns, breakdown, spans)) return;

    const auto mapping_vpn = address != nullptr ?
        address->mapping_vpn : mapping_vpn_for_lpn(lpn);
    if (config_.host.mapping_mode == MappingMode::Cached) {
        ensure_mapping_page_cached(
            mapping_vpn, at_ns, breakdown, spans, lpn, kind == MappingAccessKind::Update);
    }

    auto& mapping_dram = mapping_memory_issue_by_stack_.at(stack);
    const auto issue = reserve(
        at_ns,
        config_.host.ctrl_dram_issue_ns,
        mapping_dram);
    stats_.mapping_dram_issue_busy_ns += config_.host.ctrl_dram_issue_ns;
    if (issue.wait_ns > 0.0) {
        stats_.mapping_dram_wait_ops++;
        stats_.mapping_dram_wait_ns += issue.wait_ns;
        stats_.mapping_dram_wait_max_ns = std::max(
            stats_.mapping_dram_wait_max_ns,
            issue.wait_ns);
        breakdown.scheduler_queue_wait_ns += issue.wait_ns;
        if (spans != nullptr) {
            trace_wait(
                spans,
                mapping_entity(stack),
                at_ns,
                issue.start_ns,
                config_.host.mapping_mode == MappingMode::FullResident ?
                    "wait_resident_mapping_dram" :
                    "wait_mapping_cache_dram");
        }
    }
    const auto data_done = mapping_memory_transfer(stack,
        config_.device.page_size_bytes / config_.host.mapping_entries_per_page,
        kind == MappingAccessKind::Lookup ? Op::Read : Op::Write,
        issue.start_ns, breakdown, spans);
    const double response_done = std::max({issue.finish_ns, data_done,
        issue.start_ns + config_.host.ctrl_dram_latency_ns});
    breakdown.mapping_dram_ns += config_.host.ctrl_dram_latency_ns;
    if (spans != nullptr) {
        const auto access_name =
            config_.host.mapping_mode == MappingMode::FullResident ?
                (kind == MappingAccessKind::Lookup ?
                    "resident_mapping_lookup" :
                    "resident_mapping_update_access") :
                (kind == MappingAccessKind::Lookup ?
                    "cached_mapping_lookup" :
                    "cached_mapping_update_access");
        add_trace_span(
            spans,
            access_name,
            "metadata",
            mapping_entity(stack),
            issue.start_ns,
            response_done,
            true,
            "lpn" + std::to_string(lpn));
    }
    at_ns = response_done;
    if (kind == MappingAccessKind::Lookup) fill_mapping_sram(lpn, at_ns, breakdown, spans);
    if (config_.host.mapping_mode == MappingMode::Cached) {
        auto& cache = mapping_cache_by_stack_.at(stack);
        const auto found = cache.find(mapping_cache_key(lpn));
        if (found == cache.end()) {
            throw std::runtime_error(
                "HBF mapping-cache line disappeared during DRAM access");
        }
        found->second.evictable_ns = std::max(
            found->second.evictable_ns, response_done);
    }

    if (kind == MappingAccessKind::Update) {
        service_mapping_compute(stack, config_.host.mapping_update_ns,
            at_ns, breakdown, spans, "mapping_publish_compute");
        if (config_.host.mapping_mode == MappingMode::Cached) {
            if (config_.host.mapping_cache_layout == MappingCacheLayout::Entry) {
                if (auto* page = mapping_buffer_page(stack, mapping_vpn)) {
                    if (page->fill_ready_ns > at_ns) {
                        breakdown.scheduler_queue_wait_ns += page->fill_ready_ns - at_ns;
                        at_ns = page->fill_ready_ns;
                    }
                    mapping_dram_access(at_ns, stack, breakdown, spans,
                        "mapping_buffer_update", stats_.mapping_buffer_patch_ops, true);
                    page->evictable_ns = std::max(page->evictable_ns, at_ns);
                    page->touch = next_cache_touch_sequence_++;
                }
            }
            auto& cache = mapping_cache_by_stack_.at(stack);
            const auto found = cache.find(mapping_cache_key(lpn));
            if (found == cache.end()) {
                throw std::runtime_error(
                    "HBF mapping-cache line disappeared during update");
            }
            found->second.evictable_ns = std::max(
                found->second.evictable_ns, at_ns);
            found->second.modified_ns = at_ns;
        }
    }
}

void HbfController::flush_dirty_mapping_page(
    std::uint64_t mapping_vpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    auto events = pending_dirty_mapping_events_.find(mapping_vpn);
    if (events == pending_dirty_mapping_events_.end()) {
        clear_mapping_page_dirty(mapping_vpn);
        return;
    }
    auto snapshot = latest_touch_through(events->second, at_ns);
    if (!snapshot) {
        return;
    }
    const auto clear_snapshot = [this, mapping_vpn, &snapshot]() {
        auto pending = pending_dirty_mapping_events_.find(mapping_vpn);
        if (pending == pending_dirty_mapping_events_.end()) {
            return;
        }
        pending->second.erase(
            pending->second.begin(), pending->second.upper_bound(*snapshot));
        if (pending->second.empty()) {
            pending_dirty_mapping_events_.erase(pending);
            clear_mapping_page_dirty(mapping_vpn);
        }
    };
    const auto stack = stack_for_vpn(mapping_vpn);
    const auto preferred_plane = mapping_plane_for_vpn(mapping_vpn);
    if (!gc_active_by_stack_.at(stack)) {
        maybe_run_gc(at_ns,
            breakdown,
            spans,
            1,
            stack,
            BlockRole::Mapping,
            preferred_plane);
    }
    // Space preparation can relocate data and update this very mapping
    // page. Every layout snapshots the generation actually present when
    // the checkpoint is issued, including those completed GC updates.
    events = pending_dirty_mapping_events_.find(mapping_vpn);
    if (events == pending_dirty_mapping_events_.end()) return;
    snapshot = latest_touch_through(events->second, at_ns);
    if (!snapshot) return;
    const auto new_ppn = allocate_free_page(
        at_ns,
        breakdown,
        spans,
        BlockRole::Mapping,
        stack,
        preferred_plane);
    if (config_.host.mapping_cache_layout == MappingCacheLayout::Entry) {
        read_mapping_backing_page(mapping_vpn, at_ns, breakdown, spans, true);
        // A retained fill has been patched by all resident entry updates.
        // Its ready frontier can include work newer than the initial snapshot.
        snapshot = latest_touch_through(pending_dirty_mapping_events_.at(mapping_vpn), at_ns);
    }
    double source_consumed_ns = at_ns;
    const double program_done = schedule_program_page(
        new_ppn,
        at_ns,
        breakdown,
        spans,
        TransactionSource::Mapping,
        HeatmapTrafficSource::Mapping, &source_consumed_ns);
    if (config_.host.mapping_cache_layout == MappingCacheLayout::Entry) {
        if (auto* page = mapping_buffer_page(stack, mapping_vpn))
            page->evictable_ns = std::max(page->evictable_ns, source_consumed_ns);
    }

    const auto program_commit_sequence = schedule_media_program_commit(
        new_ppn,
        metadata_lpn(mapping_vpn),
        PageOwner::Mapping,
        program_done);
    schedule_vpn_mapping_commit(
        mapping_vpn, new_ppn, program_done, program_commit_sequence);
    at_ns = program_done;
    if (config_.host.mapping_cache_layout == MappingCacheLayout::Entry) {
        mapping_dram_access(at_ns, stack, breakdown, spans,
            "mapping_checkpoint_epoch", stats_.mapping_checkpoint_epoch_ops, true);
        auto& epoch = mapping_checkpointed_through_ns_.at(mapping_vpn);
        epoch = std::max(epoch, snapshot->first);
    }
    clear_snapshot();
    stats_.physical_write_bytes += config_.device.page_size_bytes;
    stats_.mapping_program_payload_bytes += config_.device.page_size_bytes;
    stats_.page_programs++;
    stats_.mapping_page_programs++;
}

std::optional<std::uint64_t> HbfController::lookup_lpn(
    std::uint64_t lpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    LookupIntent intent,
    const CompactReadAddress* address) {
    wait_for_lpn_dependencies(lpn, at_ns, breakdown, spans, intent, address);
    access_mapping(
        lpn,
        TransactionSource::User,
        MappingAccessKind::Lookup,
        at_ns,
        breakdown,
        spans,
        address);
    const auto visible = visible_lpn_at(lpn, at_ns, address);
    if (address != nullptr && visible != address->ppn)
        ++stats_.compact_read_run_overrides;
    return visible;
}

void HbfController::wait_for_lpn_dependencies(
    std::uint64_t lpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    LookupIntent intent,
    const CompactReadAddress* address) {
    // HBF exposes ordered memory semantics per LPN: a later-arriving access
    // waits for every earlier host write to publish. A raw erase of the
    // block holding the current copy destroys it and is waited for. A GC or
    // wear-leveling relocation of the page is waited for only by a read
    // whose current copy sits in a block whose erase has already been
    // issued: until then the old copy is intact, carries the same bytes, and
    // the erase, when it is issued later, is placed behind every read
    // already issued to that block. A full-page overwrite needs no old data
    // and never waits for a relocation.
    if (const auto mapping = lpn_to_ppn_.find(lpn);
        mapping != lpn_to_ppn_.end()) {
        wait_for_pending_block_erase(
            mapping->second, at_ns, breakdown, spans);
    } else if (const auto compact = compact_lpn_ppn(lpn, address)) {
        wait_for_pending_block_erase(
            *compact, at_ns, breakdown, spans);
    }
    const bool wait_for_relocations =
        intent == LookupIntent::ReadData && current_ppn_has_pending_transition(lpn, address);
    wait_for_prior_lpn_commit(
        lpn, at_ns, breakdown, spans, wait_for_relocations);
}

bool HbfController::write_buffer_covers(
    const WriteBufferEntry& entry,
    std::uint64_t begin,
    std::uint64_t end) const {
    return dirty_ranges_cover(entry.ranges, begin, end);
}

bool HbfController::dirty_ranges_cover(
    const std::vector<DirtyRange>& ranges,
    std::uint64_t begin,
    std::uint64_t end) const {
    std::uint64_t cursor = begin;
    for (const auto& range : ranges) {
        if (range.end <= cursor) {
            continue;
        }
        if (range.begin > cursor) {
            return false;
        }
        cursor = std::max(cursor, range.end);
        if (cursor >= end) {
            return true;
        }
    }
    return cursor >= end;
}

bool HbfController::write_buffer_full_page(const WriteBufferEntry& entry) const {
    return write_buffer_covers(entry, 0, config_.device.page_size_bytes);
}

std::uint64_t HbfController::write_buffer_covered_bytes(const WriteBufferEntry& entry) const {
    std::uint64_t bytes = 0;
    for (const auto& range : entry.ranges) {
        bytes += range.end - range.begin;
    }
    return bytes;
}

double HbfController::serve_read_from_write_buffer(
    std::uint64_t lpn,
    std::uint64_t begin,
    std::uint64_t end,
    const std::vector<DirtyRange>& ranges,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto stack = static_cast<std::uint32_t>(stack_for_lpn(lpn));
    const auto bytes = end - begin;
    std::uint64_t buffered_bytes = 0;
    for (const auto& range : ranges) {
        const auto lo = std::max(range.begin, begin);
        const auto hi = std::min(range.end, end);
        if (hi > lo) {
            buffered_bytes += hi - lo;
        }
    }
    const auto erased_bytes = bytes - buffered_bytes;
    const auto read_detail = spans == nullptr ?
        std::string{} : "lpn" + std::to_string(lpn);
    // A device-DRAM write buffer answers over HBIO: the request command goes
    // in, the assembled bytes come back out.
    const auto device_channel = stack * config_.device.channels_per_stack +
        (lpn / config_.device.stacks) % config_.device.channels_per_stack;
    if (write_buffer_on_device())
        earliest_ns = schedule_external_request_command(device_channel, earliest_ns,
            breakdown, spans, "write-buffer read", "user");
    const double dram_done = schedule_write_buffer_dram_access(
        stack,
        buffered_bytes,
        false,
        earliest_ns,
        breakdown,
        spans,
        "write_buffer_read_dram",
        read_detail);
    auto& logic_die = logic_dies_.at(stack);
    const double sram_ns = transfer_time_ns(bytes, config_.device.logic_sram_bandwidth_GBps);
    auto sram = reserve(dram_done, sram_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(stack);
        trace_wait(spans, entity, dram_done, sram.start_ns, "wait_sram");
        add_trace_span(
            spans,
            erased_bytes == 0 ?
                "write_buffer_read_hit" : "write_buffer_read_assemble",
            "sram",
            entity,
            sram.start_ns,
            sram.finish_ns,
            true,
            "lpn" + std::to_string(lpn) + " buffered=" +
                std::to_string(buffered_bytes) + "B erased=" +
                std::to_string(erased_bytes) + "B");
    }
    // Buffered data still crosses the stack's external interface (already
    // decoded SRAM content: data bytes only, no ECC pass).
    stats_.write_buffer_read_hits++;
    // Count only bytes the buffer actually holds: a read of a partially
    // buffered, never-mapped page is served here with less than full coverage.
    stats_.write_buffer_read_bytes += buffered_bytes;
    if (write_buffer_on_device())
        return schedule_external_read_egress(device_channel, bytes, sram.finish_ns,
            breakdown, spans, "write_buffer_read_hbio_out", "device-DRAM write buffer");
    return sram.finish_ns;
}

double HbfController::admit_foreground_page_read(
    std::size_t stack,
    double offered_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    auto& releases = page_read_credit_release_by_stack_.at(stack);
    const auto depth =
        static_cast<std::size_t>(config_.device.page_read_queue_depth_per_stack);
    double admitted_ns = offered_ns;
    if (releases.size() >= depth) {
        admitted_ns = std::max(admitted_ns, releases.front());
        std::pop_heap(releases.begin(), releases.end(), std::greater<double>{});
        releases.pop_back();
    }
    const double wait_ns = admitted_ns - offered_ns;
    stats_.page_read_admission_events = checked_add(
        stats_.page_read_admission_events,
        1,
        "HBF page-read admission event count");
    if (wait_ns > 0.0) {
        stats_.page_read_admission_waited_pages = checked_add(
            stats_.page_read_admission_waited_pages,
            1,
            "HBF page-read admission waited-page count");
        const double updated_wait =
            stats_.page_read_admission_wait_ns + wait_ns;
        if (!std::isfinite(updated_wait)) {
            throw std::runtime_error(
                "HBF page-read admission wait work is not finite");
        }
        stats_.page_read_admission_wait_ns = updated_wait;
        stats_.page_read_admission_max_wait_ns = std::max(
            stats_.page_read_admission_max_wait_ns,
            wait_ns);
        breakdown.scheduler_queue_wait_ns += wait_ns;
        if (spans != nullptr) {
            trace_wait(
                spans,
                logic_entity(static_cast<std::uint32_t>(stack)),
                offered_ns,
                admitted_ns,
                "wait_page_read_credit");
        }
    }
    return admitted_ns;
}

void HbfController::complete_foreground_page_read(
    std::size_t stack,
    double finish_ns) {
    if (!std::isfinite(finish_ns) || finish_ns < 0.0) {
        throw std::runtime_error(
            "HBF page-read completion frontier is invalid");
    }
    auto& releases = page_read_credit_release_by_stack_.at(stack);
    releases.push_back(finish_ns);
    std::push_heap(releases.begin(), releases.end(), std::greater<double>{});
    if (releases.size() >
        static_cast<std::size_t>(config_.device.page_read_queue_depth_per_stack)) {
        throw std::runtime_error(
            "HBF page-read admission exceeded configured per-stack depth");
    }
}

double HbfController::schedule_external_write_ingress(
    std::size_t channel,
    std::uint64_t bytes,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& detail,
    const std::string& sram_span_name) {
    auto& logic_die = logic_dies_.at(channel / config_.device.channels_per_stack);
    const double payload_ns = transfer_time_ns(bytes, physical::hbf::speed_grade(config_.device.speed_grade).payload_GBps_per_channel);

    // The top-level request command has already crossed the command port.
    // Payload may be a complete host page or one fragment of a channel
    // batch. Accumulation never transfers those bytes over HBIO again.
    auto payload = media_->transfer(channel, physical::hbf::HbfDevice::Direction::HostToDevice,
        bytes, earliest_ns, reservation_causal_watermark_ns_);
    breakdown.scheduler_queue_wait_ns += payload.wait_ns;
    breakdown.hb_io_transfer_ns += payload_ns;
    if (spans != nullptr) {
        const auto entity = channel_entity(HbfAddress{.stack = static_cast<std::uint32_t>(channel / config_.device.channels_per_stack),
            .channel = static_cast<std::uint32_t>(channel % config_.device.channels_per_stack)}) + "/hbio/rx";
        trace_wait(
            spans,
            entity,
            earliest_ns,
            payload.start_ns,
            "wait_hbio_data");
        add_trace_span(
            spans,
            "ocp_program_payload",
            "hbio",
            entity,
            payload.start_ns,
            payload.finish_ns,
            true,
            std::to_string(bytes) + "B payload " + detail);
    }

    const double ingress_ready_ns = payload.finish_ns;
    const double sram_ns = transfer_time_ns(bytes, config_.device.logic_sram_bandwidth_GBps);
    auto sram = reserve(ingress_ready_ns, sram_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(static_cast<std::uint32_t>(channel / config_.device.channels_per_stack));
        trace_wait(
            spans,
            entity,
            ingress_ready_ns,
            sram.start_ns,
            "wait_sram");
        add_trace_span(
            spans,
            sram_span_name,
            "sram",
            entity,
            sram.start_ns,
            sram.finish_ns,
            true,
            std::to_string(bytes) + "B payload " + detail);
    }
    return sram.finish_ns;
}

double HbfController::schedule_external_request_command(
    std::size_t channel,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& detail, std::string_view source) {
    const double command_ns = transfer_time_ns(
        config_.device.command_address_bytes, physical::hbf::speed_grade(config_.device.speed_grade).payload_GBps_per_channel);
    auto command = media_->transfer(channel, physical::hbf::HbfDevice::Direction::Command,
        config_.device.command_address_bytes, earliest_ns, reservation_causal_watermark_ns_);
    breakdown.scheduler_queue_wait_ns += command.wait_ns;
    breakdown.hb_io_transfer_ns += command_ns;
    if (spans != nullptr) {
        const auto entity = channel_entity(HbfAddress{.stack = static_cast<std::uint32_t>(channel / config_.device.channels_per_stack),
            .channel = static_cast<std::uint32_t>(channel % config_.device.channels_per_stack)}) + "/hbio/command";
        trace_wait(
            spans,
            entity,
            earliest_ns,
            command.start_ns,
            "wait_hbio_request");
        add_trace_span(
            spans,
            std::string(source) + "/request_hbio",
            "hbio",
            entity,
            command.start_ns,
            command.finish_ns,
            true,
            std::to_string(config_.device.command_address_bytes) + "B request " + detail);
    }
    return command.finish_ns;
}

double HbfController::schedule_write_buffer_dram_access(
    std::size_t stack,
    std::uint64_t bytes,
    bool write,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& name,
    const std::string& detail) {
    if (bytes == 0 || bytes > config_.device.page_size_bytes) {
        throw std::runtime_error(
            "HBF write-buffer DRAM access must contain 1..page_size bytes");
    }
    if (write_buffer_on_device()) {
        (write ? stats_.write_buffer_dram_write_ops : stats_.write_buffer_dram_read_ops)++;
        (write ? stats_.write_buffer_dram_write_bytes : stats_.write_buffer_dram_read_bytes) += bytes;
        return device_dram_transfer(stack, bytes, write, earliest_ns, breakdown, spans, name.c_str());
    }
    auto& logic_die = logic_dies_.at(stack);
    auto issue = reserve(
        earliest_ns,
        config_.host.ctrl_dram_issue_ns,
        logic_die.write_buffer_dram_issue);
    if (issue.wait_ns > 0.0) {
        stats_.write_buffer_dram_wait_ops = checked_add(
            stats_.write_buffer_dram_wait_ops,
            1,
            "HBF write-buffer DRAM waited accesses");
        stats_.write_buffer_dram_wait_ns += issue.wait_ns;
        stats_.write_buffer_dram_wait_max_ns = std::max(
            stats_.write_buffer_dram_wait_max_ns,
            issue.wait_ns);
        breakdown.scheduler_queue_wait_ns += issue.wait_ns;
        if (spans != nullptr) {
            trace_wait(
                spans,
                logic_entity(static_cast<std::uint32_t>(stack)),
                earliest_ns,
                issue.start_ns,
                "wait_write_buffer_dram");
        }
    }
    const auto data_done = host_memory_transfer(stack, bytes, write ? Op::Write : Op::Read,
        issue.start_ns, breakdown, spans);
    const double response_done = std::max({issue.finish_ns, data_done,
        issue.start_ns + config_.host.ctrl_dram_latency_ns});
    breakdown.write_buffer_dram_ns += config_.host.ctrl_dram_latency_ns;
    if (write) {
        stats_.write_buffer_dram_write_ops = checked_add(
            stats_.write_buffer_dram_write_ops,
            1,
            "HBF write-buffer DRAM write accesses");
        stats_.write_buffer_dram_write_bytes = checked_add(
            stats_.write_buffer_dram_write_bytes,
            bytes,
            "HBF write-buffer DRAM write bytes");
    } else {
        stats_.write_buffer_dram_read_ops = checked_add(
            stats_.write_buffer_dram_read_ops,
            1,
            "HBF write-buffer DRAM read accesses");
        stats_.write_buffer_dram_read_bytes = checked_add(
            stats_.write_buffer_dram_read_bytes,
            bytes,
            "HBF write-buffer DRAM read bytes");
    }
    if (spans != nullptr) {
        add_trace_span(
            spans,
            name,
            "controller_dram",
            logic_entity(static_cast<std::uint32_t>(stack)),
            issue.start_ns,
            response_done,
            true,
            std::to_string(bytes) + "B " + detail);
    }
    return response_done;
}

double HbfController::schedule_external_read_egress(
    std::size_t channel,
    std::uint64_t bytes,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& name,
    const std::string& detail) {
    if (bytes == 0 || bytes > config_.device.page_size_bytes) {
        throw std::runtime_error(
            "HBF external read egress must contain 1..page_size payload bytes");
    }
    bytes = ((bytes + 63) / 64) * 64;
    const double duration_ns = transfer_time_ns(
        bytes, physical::hbf::speed_grade(config_.device.speed_grade).payload_GBps_per_channel);
    auto transfer = media_->transfer(channel, physical::hbf::HbfDevice::Direction::DeviceToHost,
        bytes, earliest_ns, reservation_causal_watermark_ns_);
    breakdown.scheduler_queue_wait_ns += transfer.wait_ns;
    breakdown.hb_io_transfer_ns += duration_ns;
    if (spans != nullptr) {
        const auto entity = channel_entity(HbfAddress{.stack = static_cast<std::uint32_t>(channel / config_.device.channels_per_stack),
            .channel = static_cast<std::uint32_t>(channel % config_.device.channels_per_stack)}) + "/hbio/tx";
        trace_wait(
            spans,
            entity,
            earliest_ns,
            transfer.start_ns,
            "wait_hbio_data");
        add_trace_span(
            spans,
            name,
            "hbio",
            entity,
            transfer.start_ns,
            transfer.finish_ns,
            true,
            std::to_string(bytes) + "B payload " + detail);
    }
    return transfer.finish_ns;
}

double HbfController::stage_write_buffer_range(
    std::uint64_t lpn,
    std::uint64_t begin,
    std::uint64_t end,
    HeatmapTrafficSource heatmap_source,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto stack = static_cast<std::uint32_t>(stack_for_lpn(lpn));
    const auto bytes = end - begin;
    auto& die_write_buffer = write_buffer(stack);
    auto& die_lru = write_buffer_lru(stack);
    auto found = die_write_buffer.find(lpn);
    double admit_ns = earliest_ns;
    if (found == die_write_buffer.end()) {
        // A new entry needs a free controller-DRAM slot. Occupancy counts both live
        // entries and flushed pages whose programs are still in flight; at
        // capacity the admit waits for the oldest program to complete (the
        // real backpressure of a sustained overload). An overwrite of a live
        // entry consumes no new slot and never waits here.
        auto& slot_releases = write_buffer_slot_release_by_stack_.at(stack);
        const auto release_completed_slots = [&slot_releases](double through_ns) {
            slot_releases.erase(slot_releases.begin(), slot_releases.upper_bound(through_ns));
        };
        const auto slot_capacity_reached = [&]() {
            const auto live = static_cast<std::uint64_t>(die_write_buffer.size());
            const auto inflight = static_cast<std::uint64_t>(slot_releases.size());
            if (live > config_.host.write_buffer_pages ||
                inflight > config_.host.write_buffer_pages - live) {
                throw std::runtime_error("HBF write-buffer slot accounting exceeded capacity");
            }
            return live + inflight == config_.host.write_buffer_pages;
        };

        release_completed_slots(admit_ns);
        while (slot_capacity_reached()) {
            if (!slot_releases.empty()) {
                admit_ns = std::max(admit_ns, *slot_releases.begin());
                release_completed_slots(admit_ns);
                continue;
            }
            if (die_lru.empty()) {
                throw std::runtime_error(
                    "HBF write-buffer capacity is full but has no live or in-flight owner");
            }

            // Every occupied slot is still live. Evict the LRU entry first;
            // flush_write_buffer_entry removes it from the live buffer and
            // records the media-completion release. Only after that release
            // is reached may the new LPN be installed below.
            const auto victim_lpn = die_lru.back();
            double eviction_done_ns = admit_ns;
            flush_write_buffer_entry(
                victim_lpn, eviction_done_ns, breakdown, spans);
            admit_ns = std::max(admit_ns, eviction_done_ns);
            release_completed_slots(admit_ns);
        }
        if (admit_ns > earliest_ns) {
            stats_.write_buffer_slot_wait_ops++;
            stats_.write_buffer_slot_wait_ns += admit_ns - earliest_ns;
            breakdown.scheduler_queue_wait_ns += admit_ns - earliest_ns;
            if (spans != nullptr) {
                trace_wait(
                    spans,
                    logic_entity(stack),
                    earliest_ns,
                    admit_ns,
                    "wait_write_buffer_slot");
            }
        }
        if (slot_capacity_reached()) {
            throw std::runtime_error("HBF write-buffer admission did not release a slot");
        }
        die_lru.push_front(lpn);
        found = die_write_buffer.emplace(lpn, WriteBufferEntry{
            .lpn = lpn,
            .ranges = {},
            .iterator = die_lru.begin(),
            .ready_ns = 0.0,
        }).first;
        stats_.write_buffer_misses++;
    } else {
        die_lru.splice(die_lru.begin(), die_lru, found->second.iterator);
        found->second.iterator = die_lru.begin();
        stats_.write_buffer_hits++;
    }

    const auto overlapped = insert_merged_range(
        found->second.ranges,
        DirtyRange{
            .begin = begin,
            .end = end,
            .heatmap_source = heatmap_source,
        });
    stats_.write_buffer_merged_bytes += overlapped;

    // This is the one external payload crossing for a coalesced write. The
    // logic-die SRAM is only an ingress datapath; durable-in-controller
    // buffering is charged to and written into the shared controller DRAM.
    // Later destage must not charge a second full-page HBIO transfer.
    const auto stage_detail = spans == nullptr ? std::string{} :
        "buffered lpn" + std::to_string(lpn) + " dirty=" +
            std::to_string(write_buffer_covered_bytes(found->second)) + "/" +
            std::to_string(config_.device.page_size_bytes) + "B";
    // A device-DRAM write buffer receives the payload over HBIO now; the
    // flush then programs it without a second crossing.
    const double ingress_done_ns = write_buffer_on_device() ?
        device_dram_ingress(stack, lpn / config_.device.stacks, bytes, admit_ns, breakdown, spans) :
        admit_ns;
    (void)stage_detail;
    const auto dram_detail = spans == nullptr ?
        std::string{} : "lpn" + std::to_string(lpn);
    const double stage_done_ns = schedule_write_buffer_dram_access(
        stack,
        bytes,
        true,
        ingress_done_ns,
        breakdown,
        spans,
        "write_buffer_stage_dram",
        dram_detail);
    found->second.ready_ns = std::max(found->second.ready_ns, stage_done_ns);
    return stage_done_ns;
}

void HbfController::flush_write_buffer_entry(
    std::uint64_t lpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto stack = static_cast<std::uint32_t>(stack_for_lpn(lpn));
    auto& die_write_buffer = write_buffer(stack);
    auto found = die_write_buffer.find(lpn);
    if (found == die_write_buffer.end()) {
        return;
    }
    const auto entry = found->second;
    // NAND programs are atomic at page granularity, while buffered writes may
    // combine byte ranges from several request classes. Preserve byte-level
    // last-writer provenance in the entry, then conservatively label the one
    // physical page operation with the source owning the most dirty bytes.
    // Equal-byte ties resolve by HeatmapTrafficSource enum order. This keeps
    // physical byte/access conservation exact without relabeling a deferred
    // flush from whichever request (or drain) happens to trigger it.
    const auto entry_heatmap_source = dominant_dirty_source(entry.ranges);
    at_ns = std::max(at_ns, entry.ready_ns);
    const auto flush_detail = spans == nullptr ?
        std::string{} : "lpn" + std::to_string(lpn);
    const bool full_page = write_buffer_full_page(entry);
    if (!write_buffer_on_device()) {
        at_ns = schedule_write_buffer_dram_access(
            stack,
            write_buffer_covered_bytes(entry),
            false,
            at_ns,
            breakdown,
            spans,
            "write_buffer_flush_dram",
            flush_detail);
        // This read supplies a full buffered page directly. Partial pages
        // require the assembly writes/read below before the single device
        // ingress. A device-DRAM write buffer instead assembles the page in
        // place and reads it once for the program, below.
    }

    // Resolve the mapping at flush time: GC may have relocated the page since
    // the write was staged. State-only read; translation time was charged at stage.
    wait_for_lpn_dependencies(
        lpn, at_ns, breakdown, spans,
        full_page ? LookupIntent::OverwriteFullPage : LookupIntent::ReadData);
    const auto current_ppn = visible_lpn_at(lpn, at_ns);

    if (current_ppn && !full_page) {
        // The old media image is read and merged whether or not the run is
        // traced: tracing must never gate timing or accounting work.
        if (spans != nullptr) {
            add_trace_span(
                spans,
                "partial_page_merge_on_flush",
                "translation",
                logic_entity(stack),
                at_ns,
                at_ns + config_.host.mapping_update_ns,
                true,
                "lpn" + std::to_string(lpn));
        }
        breakdown.translation_ns += config_.host.mapping_update_ns;
        at_ns += config_.host.mapping_update_ns;
        if (write_buffer_on_device()) {
            // The old image merges inside device DRAM: a cached copy is read
            // there, otherwise NAND decodes it into DRAM without leaving the
            // device. This version is superseded by the flush, so a decoded
            // copy is not kept as a read-cache line.
            if (const auto cached = device_dram_cached(*current_ppn)) {
                at_ns = device_dram_transfer(stack, config_.device.page_size_bytes, false,
                    std::max(at_ns, *cached), breakdown, spans, "write_buffer_merge_device_dram");
            } else {
                at_ns = device_dram_transfer(stack, config_.device.page_size_bytes, true,
                    schedule_read_page(*current_ppn, at_ns, breakdown, spans,
                        TransactionSource::User, entry_heatmap_source,
                        ReadPayloadRoute::DeviceDram, 0),
                    breakdown, spans, "write_buffer_merge_device_dram");
                stats_.physical_read_bytes += config_.device.page_size_bytes;
                stats_.page_reads++;
            }
        } else {
            at_ns = schedule_read_page(
                *current_ppn,
                at_ns,
                breakdown,
                spans,
                TransactionSource::User,
                entry_heatmap_source,
                ReadPayloadRoute::HostBuffer,
                0);
            stats_.physical_read_bytes += config_.device.page_size_bytes;
            stats_.page_reads++;
        }
    }
    if (!full_page) {
        const auto assembled_bytes =
            current_ppn ? write_buffer_covered_bytes(entry) : config_.device.page_size_bytes;
        if (write_buffer_on_device()) {
            // The dirty bytes (or a zero-filled page) are written over the
            // old image in place.
            at_ns = device_dram_transfer(stack, assembled_bytes, true, at_ns, breakdown, spans,
                "write_buffer_assemble_device_dram");
        } else {
            at_ns = host_memory_transfer(stack, assembled_bytes, Op::Write, at_ns, breakdown, spans);
            at_ns = host_memory_transfer(stack, config_.device.page_size_bytes,
                Op::Read, at_ns, breakdown, spans);
        }
    }
    if (write_buffer_on_device()) {
        // The program reads the assembled page from device DRAM once, for a
        // full and for a partial page alike (the structural policy's
        // program() charges the same read).
        at_ns = schedule_write_buffer_dram_access(
            stack,
            config_.device.page_size_bytes,
            false,
            at_ns,
            breakdown,
            spans,
            "write_buffer_flush_dram",
            flush_detail);
    }

    maybe_run_gc(at_ns,
        breakdown,
        spans,
        1,
        stack_for_lpn(lpn),
        BlockRole::Data);
    const auto new_ppn = allocate_free_page(
        at_ns,
        breakdown,
        spans,
        BlockRole::Data,
        stack_for_lpn(lpn));

    program_payload_on_device_ = write_buffer_on_device();
    const double program_done = schedule_program_page(
        new_ppn,
        at_ns,
        breakdown,
        spans,
        TransactionSource::User,
        entry_heatmap_source);
    program_payload_on_device_ = false;
    const auto program_commit_sequence = schedule_media_program_commit(
        new_ppn,
        lpn,
        PageOwner::Logical,
        program_done);
    double mapping_ready = program_done;
    access_mapping(
        lpn,
        TransactionSource::User,
        MappingAccessKind::Update,
        mapping_ready,
        breakdown,
        spans);
    const double page_done = mapping_ready;
    schedule_lpn_mapping_commit(
        lpn, new_ppn, page_done, program_commit_sequence);
    mark_mapping_page_dirty(mapping_vpn_for_lpn(lpn), page_done);
    if (write_buffer_on_device()) {
        // Write-allocate: the flushed page becomes a clean read-cache line
        // when its write-buffer slot is released (below, at page_done), so
        // the page never holds a slot and a line at once. The program has
        // materialized by then; a page retired before that (TRIM, erase)
        // is not cached.
        const auto block = static_cast<std::size_t>(new_ppn / config_.device.pages_per_block);
        const auto epoch = blocks_.at(block).epoch;
        (void)schedule_commit(stack, page_done, [this, new_ppn, block, epoch, page_done]() {
            const auto& state = blocks_.at(block);
            if (state.epoch != epoch ||
                !state.is_valid(static_cast<std::uint32_t>(new_ppn % config_.device.pages_per_block)))
                return;
            device_dram_install(new_ppn, page_done);
        });
    }
    at_ns = page_done;
    // The dirty resident entry persists through its checkpoint at drain.

    const auto generation = next_buffer_generation_++;
    inflight_buffered_writes_[lpn].push_back(InflightBufferedWrite{
        .generation = generation,
        .ranges = entry.ranges,
        .ready_ns = entry.ready_ns,
        .commit_ns = page_done,
        .target_ppn = new_ppn,
        .target_block_epoch = blocks_.at(
            static_cast<std::size_t>(new_ppn / config_.device.pages_per_block)).epoch,
    });
    (void)schedule_commit(stack, page_done, [this, lpn, generation]() {
        const auto found_inflight = inflight_buffered_writes_.find(lpn);
        if (found_inflight == inflight_buffered_writes_.end()) {
            return;
        }
        std::erase_if(
            found_inflight->second,
            [generation](const InflightBufferedWrite& write) {
                return write.generation == generation;
            });
        if (found_inflight->second.empty()) {
            inflight_buffered_writes_.erase(found_inflight);
        }
    });

    write_buffer_lru(stack).erase(entry.iterator);
    die_write_buffer.erase(found);
    // The generation remains readable from controller DRAM until the logical mapping
    // commits. Releasing its physical slot at program_done allowed a new LPN
    // to reuse the same one-page buffer while the old generation was still
    // reported as a DRAM hit. Keep ownership and readability on one deadline.
    write_buffer_slot_release_by_stack_.at(stack).insert(page_done);
    stats_.physical_write_bytes += config_.device.page_size_bytes;
    stats_.data_program_payload_bytes += config_.device.page_size_bytes;
    stats_.data_programs++;
    stats_.page_programs++;
    stats_.write_buffer_flushes++;
}

void HbfController::flush_all_write_buffer_entries(
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    // Drain fans out: every entry's flush is issued from the drain start and
    // the media resource reservations do the real serialization; a single
    // chained cursor would serialize the issue times themselves.
    const double drain_start_ns = at_ns;
    double drain_finish_ns = at_ns;
    for (std::size_t stack = 0; stack < config_.device.stacks; ++stack) {
        auto& die_lru = write_buffer_lru(stack);
        while (!die_lru.empty()) {
            const auto victim_lpn = die_lru.back();
            double entry_ns = drain_start_ns;
            flush_write_buffer_entry(victim_lpn, entry_ns, breakdown, spans);
            drain_finish_ns = std::max(drain_finish_ns, entry_ns);
        }
    }
    at_ns = drain_finish_ns;
}

std::uint64_t HbfController::allocate_free_page(
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    BlockRole role,
    std::size_t stack,
    std::optional<std::size_t> preferred_plane) {
    if (free_pages_per_stack_.at(stack) == 0) {
        throw std::runtime_error(
            "HBF stack " + std::to_string(stack) +
            " free-page pool exhausted before GC could free space (shared-nothing "
            "FTL: stacks cannot borrow pages from each other)");
    }
    breakdown.address_mapping_ns += config_.host.free_page_allocation_ns;
    if (spans != nullptr) {
        add_trace_span(
            spans,
            "free_page_alloc",
            "translation",
            "logic/free_page_allocator",
            at_ns,
            at_ns + config_.host.free_page_allocation_ns,
            true,
            role_name(role));
    }
    at_ns += config_.host.free_page_allocation_ns;

    const auto take_page = [&](std::size_t plane_candidate) {
        auto& active = active_block_ref(planes_.at(plane_candidate), role);
        const auto ppn = allocate_page_from_block(*active);
        if (blocks_.at(*active).free_pages == 0) {
            active = std::nullopt;
        }
        return ppn;
    };
    const auto active_has_room = [&](std::size_t plane_candidate) {
        auto& active = active_block_ref(planes_.at(plane_candidate), role);
        if (active && blocks_.at(*active).free_pages != 0) {
            return true;
        }
        if (active) mark_gc_headroom_dirty(plane_candidate);
        active = std::nullopt;
        return false;
    };
    const auto pps = planes_per_stack();
    const auto stack_base = stack * pps;
    if (preferred_plane &&
        (*preferred_plane < stack_base || *preferred_plane >= stack_base + pps)) {
        throw std::runtime_error(
            "HBF preferred allocation plane belongs to another stack");
    }

    if (role == BlockRole::GC) {
        // Relocation fills the stack's open GC frontier wherever it is
        // before opening another block, so at most one GC block per stack
        // holds free pages and the reserve is consumed one block at a time.
        // A new frontier opens in the victim's plane when it has a free
        // block (host allocation locality), else in the plane with the deepest
        // free pool; GC may open reserved blocks.
        if (preferred_plane && active_has_room(*preferred_plane)) {
            return take_page(*preferred_plane);
        }
        for (std::size_t plane_candidate = stack_base;
             plane_candidate < stack_base + pps;
             ++plane_candidate) {
            if (active_has_room(plane_candidate)) {
                return take_page(plane_candidate);
            }
        }
        std::optional<std::size_t> open_plane;
        if (preferred_plane && !planes_.at(*preferred_plane).free_blocks.empty()) {
            open_plane = preferred_plane;
        } else {
            for (std::size_t plane_candidate = stack_base;
                 plane_candidate < stack_base + pps;
                 ++plane_candidate) {
                const auto depth = planes_.at(plane_candidate).free_blocks.size();
                if (depth != 0 &&
                    (!open_plane ||
                     depth > planes_.at(*open_plane).free_blocks.size())) {
                    open_plane = plane_candidate;
                }
            }
        }
        if (open_plane) {
            active_block_ref(planes_.at(*open_plane), role) =
                allocate_block_from_plane(*open_plane, role);
            return take_page(*open_plane);
        }
    } else {
        // Give each selected plane its own frontier before falling back to
        // another plane's open block. Scanning all open blocks first pinned
        // block-aligned populations to one plane and lost write parallelism.
        // Opening a frontier still preserves the stack's GC reserve.
        if (preferred_plane && active_has_room(*preferred_plane)) {
            return take_page(*preferred_plane);
        }
        const auto first = choose_allocation_plane(stack, role);
        const auto open_frontier = [&](std::size_t plane_candidate) {
            active_block_ref(planes_.at(plane_candidate), role) =
                allocate_block_from_plane(plane_candidate, role);
            return take_page(plane_candidate);
        };
        std::optional<bool> may_open;
        const auto try_plane = [&](std::size_t plane_candidate)
            -> std::optional<std::uint64_t> {
            if (active_has_room(plane_candidate)) {
                return take_page(plane_candidate);
            }
            if (planes_.at(plane_candidate).free_blocks.empty()) {
                return std::nullopt;
            }
            if (!may_open) {
                const auto pool = gc_headroom(stack, BlockRole::Data).relocation_pages;
                may_open = pool >= checked_add(
                    gc_reserve_requirement_pages(), config_.device.pages_per_block,
                    "HBF foreground block admission");
            }
            if (!*may_open) {
                return std::nullopt;
            }
            return open_frontier(plane_candidate);
        };
        if (preferred_plane) {
            if (const auto page = try_plane(*preferred_plane)) {
                return *page;
            }
        }
        for (std::size_t offset = 0; offset < pps; ++offset) {
            const auto plane_candidate = stack_base + (first + offset) % pps;
            if (const auto page = try_plane(plane_candidate)) {
                return *page;
            }
        }
        // A GC checkpoint may borrow reserve only after every existing
        // mapping frontier is exhausted. Opening extra frontiers from this
        // pool would consume the live victim's remaining relocation space.
        if (role == BlockRole::Mapping && gc_active_by_stack_.at(stack)) {
            if (preferred_plane && !planes_.at(*preferred_plane).free_blocks.empty()) {
                return open_frontier(*preferred_plane);
            }
            for (std::size_t offset = 0; offset < pps; ++offset) {
                const auto plane_candidate = stack_base + (first + offset) % pps;
                if (!planes_.at(plane_candidate).free_blocks.empty()) {
                    return open_frontier(plane_candidate);
                }
            }
        }
    }

    throw std::runtime_error(
        "HBF cannot allocate a page for role " + role_name(role) +
        ": free pages exist only in the GC-only reserve or other roles' active "
        "blocks (relocation_capacity=" +
        std::to_string(gc_relocation_capacity(stack)) +
        " pages; the foreground blocks until GC returns a block, or automatic "
        "GC is disabled)");
}

std::size_t HbfController::choose_allocation_plane(
    std::size_t stack,
    BlockRole role) {
    const auto pps = planes_per_stack();
    if (pps == 0) {
        throw std::runtime_error("HBF has no planes");
    }
    std::vector<std::size_t>* cursors = nullptr;
    switch (role) {
    case BlockRole::Data:
        cursors = &next_data_allocation_plane_per_stack_;
        break;
    case BlockRole::Mapping:
        cursors = &next_mapping_allocation_plane_per_stack_;
        break;
    case BlockRole::GC:
    case BlockRole::Free:
    case BlockRole::StaticReadOnly:
    case BlockRole::RawPhysical:
        throw std::runtime_error(
            "HBF foreground allocation cursor requires a Data or Mapping role");
    }
    auto& cursor = cursors->at(stack);
    const auto selected = cursor % pps;
    cursor = (cursor + 1) % pps;
    return selected;
}

std::size_t HbfController::allocate_block_from_plane(std::size_t plane_index, BlockRole role) {
    auto& plane = planes_.at(plane_index);
    if (plane.free_blocks.empty()) {
        throw std::runtime_error("HBF plane free-block pool exhausted");
    }
    const auto block_index = plane.free_blocks.front();
    mark_gc_headroom_dirty(plane_index);
    plane.free_blocks.pop_front();
    auto& block = blocks_.at(block_index);
    if (block.role != BlockRole::Free || block.free_pages != config_.device.pages_per_block ||
        block.valid_pages != 0 || block.invalid_pages != 0 || block.next_page != 0 ||
        block.pending_program_pages != 0 || block.erase_pending ||
        block.pending_mapping_publications != 0) {
        throw std::runtime_error("HBF free-block pool contains a non-erased block");
    }
    set_block_role(block_index, role);
    return block_index;
}

std::optional<std::size_t>& HbfController::active_block_ref(PlaneState& plane, BlockRole role) {
    switch (role) {
    case BlockRole::Data:
        return plane.active_data_block;
    case BlockRole::Mapping:
        return plane.active_mapping_block;
    case BlockRole::GC:
        return plane.active_gc_block;
    case BlockRole::RawPhysical:
        throw std::runtime_error("HBF FTL cannot allocate a page from a raw-physical block");
    case BlockRole::StaticReadOnly:
        throw std::runtime_error("HBF cannot allocate a page from a static-data block");
    case BlockRole::Free:
        throw std::runtime_error("HBF cannot allocate a page from a free-role block");
    }
    throw std::runtime_error("unknown HBF block role");
}

std::uint64_t HbfController::allocate_page_from_block(std::size_t block_index) {
    auto& block = blocks_.at(block_index);
    if (block.role == BlockRole::Free || block.free_pages == 0 ||
        block.next_page >= config_.device.pages_per_block) {
        throw std::runtime_error("HBF allocator selected a block with no programmable pages");
    }
    const auto page = block.next_page++;
    block.free_pages--;
    mark_gc_headroom_dirty(block_plane_index(block_index));
    block.pending_program_pages++;
    free_pages_--;
    free_pages_per_stack_.at(stack_of_block(block_index))--;
    const auto ppn = static_cast<std::uint64_t>(block_index) * config_.device.pages_per_block + page;
    const auto inserted = programmed_pages_.emplace(ppn, PageState{});
    if (!inserted.second && inserted.first->second.status != PageStatus::Erased) {
        throw std::runtime_error("HBF allocator selected a programmed physical page");
    }
    inserted.first->second.status = PageStatus::Erased;
    inserted.first->second.owner = PageOwner::Unassigned;
    inserted.first->second.lpn = 0;
    inserted.first->second.block_epoch = block.epoch;
    mark_gc_candidate_dirty(block_index);
    return ppn;
}

std::uint64_t HbfController::allocate_compact_pages_on_plane(
    std::size_t plane_index,
    BlockRole role,
    std::uint32_t page_count,
    std::vector<std::uint64_t>* assigned_blocks,
    std::unordered_map<std::uint64_t, std::uint32_t>*
        live_pages_by_block) {
    if (page_count == 0 ||
        (role != BlockRole::Data && role != BlockRole::Mapping)) {
        throw std::runtime_error(
            "HBF compact allocation requires positive Data/Mapping pages");
    }
    auto& plane = planes_.at(plane_index);
    std::optional<std::uint64_t> first_ppn;
    auto remaining = page_count;
    while (remaining != 0) {
        auto& active = active_block_ref(plane, role);
        if (!active || blocks_.at(*active).free_pages == 0) {
            if (active) mark_gc_headroom_dirty(plane_index);
            active.reset();
            const auto stack = stack_of_plane(plane_index);
            if (plane.free_blocks.empty() ||
                gc_headroom(stack, BlockRole::Data).relocation_pages <
                    checked_add(
                        gc_reserve_requirement_pages(), config_.device.pages_per_block,
                        "HBF compact image block admission")) {
                throw std::runtime_error(
                    "HBF compact image exhausted an allocatable plane block");
            }
            active = allocate_block_from_plane(plane_index, role);
            if (assigned_blocks != nullptr) {
                assigned_blocks->push_back(*active);
            }
        }
        auto& block = blocks_.at(*active);
        if (assigned_blocks != nullptr &&
            (assigned_blocks->empty() || assigned_blocks->back() != *active)) {
            throw std::runtime_error(
                "HBF compact data-block directory lost allocator order");
        }
        const auto take = std::min<std::uint32_t>(remaining, block.free_pages);
        const auto first_page = block.next_page;
        const auto ppn =
            static_cast<std::uint64_t>(*active) * config_.device.pages_per_block +
            first_page;
        if (!first_ppn) {
            first_ppn = ppn;
        }
        block.set_valid_range(first_page, take, config_.device.pages_per_block);
        block.next_page += take;
        block.free_pages -= take;
        mark_gc_headroom_dirty(plane_index);
        block.valid_pages += take;
        if (live_pages_by_block != nullptr) {
            auto& live = (*live_pages_by_block)[*active];
            if (take > std::numeric_limits<std::uint32_t>::max() - live) {
                throw std::runtime_error(
                    "HBF compact per-block live-page count overflowed");
            }
            live += take;
        }
        free_pages_ -= take;
        free_pages_per_stack_.at(stack_of_block(*active)) -= take;
        remaining -= take;
        mark_gc_candidate_dirty(*active);
        if (block.free_pages == 0) {
            active.reset();
        }
    }
    return *first_ppn;
}

void HbfController::invalidate_ppn(std::uint64_t ppn) {
    auto found = programmed_pages_.find(ppn);
    if (found == programmed_pages_.end() || found->second.status != PageStatus::Valid) {
        return;
    }
    invalidate_live_page(ppn);
    // Publication has retired this version. Delayed operations keep their
    // own PPN/epoch and ownership pins; no reader or relocation needs the old
    // logical identity. Erasing metadata does NOT free physical capacity.
    programmed_pages_.erase(found);
}

void HbfController::invalidate_live_page(std::uint64_t ppn) {
    auto& block = blocks_.at(ppn / config_.device.pages_per_block);
    const auto page = static_cast<std::uint32_t>(ppn % config_.device.pages_per_block);
    if (block.valid_pages == 0 || !block.is_valid(page)) {
        // A valid page always counts in its block; an underflow here means
        // the page table and the block counters disagree, which the audit
        // exists to catch rather than hide.
        throw std::runtime_error(
            "HBF invalidated a page in a block with no valid pages");
    }
    block.valid_pages--;
    block.clear_valid(page);
    if (block.valid_pages == 0) block.valid_bitmap.reset();
    // A dead version never serves a read again: TRIM, overwrite publication
    // and relocation all retire its device-DRAM line here.
    device_dram_purge_page(ppn);
    block.invalid_pages++;
    mark_gc_candidate_dirty(static_cast<std::size_t>(ppn / config_.device.pages_per_block));
    stats_.invalidations++;
}

void HbfController::mark_programmed(
    std::uint64_t ppn,
    std::uint64_t lpn,
    PageOwner owner) {
    if (ppn >= total_pages_) {
        throw std::runtime_error("HBF program target PPN is out of range");
    }
    const auto block_index = static_cast<std::size_t>(ppn / config_.device.pages_per_block);
    const auto page_index = static_cast<std::uint32_t>(ppn % config_.device.pages_per_block);
    auto& block = blocks_.at(block_index);
    auto page_state = programmed_pages_.find(ppn);
    if (page_state == programmed_pages_.end()) {
        throw std::runtime_error(
            "HBF program completion has no allocator/raw reservation");
    }
    if (page_state->second.block_epoch != block.epoch || block.erase_pending) {
        throw std::runtime_error("HBF program completion targets a retired block epoch");
    }
    const bool role_matches =
        (owner == PageOwner::Logical &&
            (block.role == BlockRole::Data || block.role == BlockRole::GC)) ||
        (owner == PageOwner::Mapping &&
            (block.role == BlockRole::Mapping || block.role == BlockRole::GC)) ||
        (owner == PageOwner::RawPhysical && block.role == BlockRole::RawPhysical);
    if (!role_matches) {
        throw std::runtime_error("HBF program owner does not match block role");
    }
    if (page_state->second.status != PageStatus::Erased) {
        throw std::runtime_error("HBF attempted to program a non-erased physical page");
    }
    if (block.pending_program_pages == 0) {
        throw std::runtime_error(
            "HBF program completion lost its block ownership pin");
    }
    // The block epoch changes on erase, but not when an erased page becomes
    // programmed. Retire any decoded erased-value cache line exactly when the
    // media program completes so a later physical read cannot observe the old
    // contents under the still-current block epoch.
    read_buffer_purge_page(ppn);
    page_state->second.status = PageStatus::Valid;
    page_state->second.owner = owner;
    page_state->second.lpn = lpn;
    block.pending_program_pages--;
    block.valid_pages++;
    block.set_valid(page_index, config_.device.pages_per_block);
    mark_gc_candidate_dirty(block_index);
}

void HbfController::release_invalid_block(std::size_t block_index) {
    const auto& retiring_block = blocks_.at(block_index);
    if (!retiring_block.erase_pending &&
        (retiring_block.pending_program_pages != 0 ||
         retiring_block.pending_mapping_publications != 0)) {
        throw std::runtime_error(
            "HBF GC attempted to reclaim a block with in-flight ownership");
    }
    read_buffer_purge_block(block_index);
    device_dram_purge_block(block_index);
    const auto block_begin = static_cast<std::uint64_t>(block_index) * config_.device.pages_per_block;
    for (std::uint32_t page = 0; page < config_.device.pages_per_block; ++page) {
        programmed_pages_.erase(block_begin + page);
    }

    auto& block = blocks_.at(block_index);
    const bool was_free = block.role == BlockRole::Free;
    const bool was_pending_free_erase = was_free && block.erase_pending;
    const auto next_epoch = block.erase_pending ? block.epoch : block.epoch + 1;
    const auto newly_free_pages = config_.device.pages_per_block - block.free_pages;
    free_pages_ += newly_free_pages;
    free_pages_per_stack_.at(stack_of_block(block_index)) += newly_free_pages;
    // The P/E cycle was counted when the erase was scheduled.
    const auto erase_count = block.erase_count;
    const bool was_unmanaged = block.role == BlockRole::RawPhysical || block.role == BlockRole::StaticReadOnly;
    const auto gc_bucket = block.gc_bucket;
    const auto gc_erase_key = block.gc_erase_key;
    const auto gc_index_dirty = block.gc_index_dirty;
    set_block_role(block_index, BlockRole::Free);
    block = BlockState{};
    block.free_pages = config_.device.pages_per_block;
    block.erase_count = erase_count;
    block.epoch = next_epoch;
    block.gc_bucket = gc_bucket;
    block.gc_erase_key = gc_erase_key;
    block.gc_index_dirty = gc_index_dirty;
    mark_gc_candidate_dirty(block_index);
    if (was_unmanaged) add_managed_block_wear(block_index);

    auto& plane = planes_.at(block_plane_index(block_index));
    if (plane.active_data_block == block_index) {
        plane.active_data_block = std::nullopt;
    }
    if (plane.active_mapping_block == block_index) {
        plane.active_mapping_block = std::nullopt;
    }
    if (plane.active_gc_block == block_index) {
        plane.active_gc_block = std::nullopt;
    }
    if (!was_free || was_pending_free_erase) {
        plane.free_blocks.push_back(block_index);
    }
}

HbfController::ScheduledTransfer HbfController::reserve(
    double earliest_ns,
    double duration_ns,
    ResourceTimeline& timeline) {
    // Validate the complete reservation before pruning mutates the calendar.
    // Legal internal work is never ready before its top-level issue/drain
    // arrival; violating that invariant must fail without consuming history.
    if (!std::isfinite(reservation_causal_watermark_ns_) ||
        reservation_causal_watermark_ns_ < 0.0 ||
        !std::isfinite(earliest_ns) ||
        earliest_ns < reservation_causal_watermark_ns_ ||
        !std::isfinite(duration_ns) || duration_ns <= 0.0) {
        throw std::runtime_error("HBF resource calendar received an invalid reservation");
    }
    const double updated_reserved_work_ns =
        timeline.reserved_work_ns + duration_ns;
    if (!std::isfinite(updated_reserved_work_ns)) {
        throw std::runtime_error(
            "HBF resource reservation work exceeds finite double range");
    }
    timeline.prune_before(reservation_causal_watermark_ns_);
    const auto reserved = timeline.reserve(earliest_ns, duration_ns);
    return ScheduledTransfer{
        .start_ns = reserved.start_ns,
        .finish_ns = reserved.finish_ns,
        .wait_ns = std::max(0.0, reserved.start_ns - earliest_ns),
    };
}

HbfController::ScheduledTransfer HbfController::schedule_flash_transaction(
    const HbfAddress& addr,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    TransactionKind kind) {
    auto& die = dies_.at(die_index(addr));
    const auto source_slot = source_index(source);
    stats_.flash_scheduler_enqueues++;

    auto slot = reserve(
        earliest_ns, config_.device.flash_tsu_issue_ns, die.source_queues.at(source_slot));
    breakdown.scheduler_queue_wait_ns += slot.wait_ns;
    const double source_ready = slot.start_ns;

    auto issue = reserve(source_ready, config_.device.flash_tsu_issue_ns, die.sequencer);
    breakdown.scheduler_queue_wait_ns += issue.wait_ns;
    breakdown.command_ns += config_.device.flash_tsu_issue_ns;
    if (spans != nullptr) {
        const auto source_label = source_name(source);
        const auto kind_label = kind_name(kind);
        trace_wait(
            spans,
            die_entity(addr),
            earliest_ns,
            slot.start_ns,
            "wait_flash_scheduler_" + source_label);
        trace_wait(
            spans,
            die_entity(addr),
            source_ready,
            issue.start_ns,
            "wait_flash_sequencer");
        add_trace_span(
            spans,
            source_label + "/flash_scheduler_issue_" + kind_label,
            "sequencer",
            die_entity(addr),
            issue.start_ns,
            issue.finish_ns,
            true,
            source_label + " " + kind_label);
    }

    die.sequencer_busy_ns += config_.device.flash_tsu_issue_ns;
    die.transaction_count++;
    stats_.flash_scheduler_issues++;
    return issue;
}

double HbfController::schedule_command_path(
    const HbfAddress& addr,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source, bool host_command) {
    if (host_command) earliest_ns = schedule_external_request_command(channel_index(addr),
        earliest_ns, breakdown, spans, "OCP page command", source_name(source));
    auto& logic_die = logic_dies_.at(stack_index(addr));
    auto& channel = channels_.at(channel_index(addr));
    const double tsv_ns = transfer_time_ns(config_.device.command_address_bytes, config_.device.tsv_bandwidth_GBps);
    const double channel_ns = transfer_time_ns(config_.device.command_address_bytes, config_.device.channel_bandwidth_GBps);

    breakdown.tsv_transfer_ns += tsv_ns;
    auto tsv = reserve(earliest_ns, tsv_ns, logic_die.tsv);
    breakdown.scheduler_queue_wait_ns += tsv.wait_ns;

    auto command = reserve(tsv.finish_ns, channel_ns, channel.command);
    breakdown.scheduler_queue_wait_ns += command.wait_ns;
    breakdown.channel_transfer_ns += channel_ns;
    channel.command_busy_ns += channel_ns;
    channel.command_count++;
    if (spans != nullptr) {
        const auto source_label = source_name(source);
        const auto tsv_entity =
            "stack" + std::to_string(addr.stack) + "/tsv";
        trace_wait(
            spans,
            tsv_entity,
            earliest_ns,
            tsv.start_ns,
            "wait_tsv_cmd");
        add_trace_span(
            spans,
            source_label + "/cmd_addr_tsv",
            "tsv",
            tsv_entity,
            tsv.start_ns,
            tsv.finish_ns,
            true,
            std::to_string(config_.device.command_address_bytes) + "B " + source_label);
        trace_wait(
            spans,
            channel_entity(addr),
            tsv.finish_ns,
            command.start_ns,
            "wait_channel_cmd");
        add_trace_span(
            spans,
            source_label + "/cmd_addr_channel",
            "flash_channel",
            channel_entity(addr),
            command.start_ns,
            command.finish_ns,
            true,
            std::to_string(config_.device.command_address_bytes) + "B " + source_label);
    }
    return command.finish_ns;
}

double HbfController::schedule_ecc(
    const HbfAddress& addr,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    bool decode) {
    auto& die = dies_.at(die_index(addr));
    const double latency_ns = decode ?
        config_.device.ecc_decode_latency_ns : config_.device.ecc_encode_latency_ns;
    const double raw_bandwidth_GBps = decode ?
        config_.device.ecc_decode_raw_bandwidth_GBps_per_die :
        config_.device.ecc_encode_raw_bandwidth_GBps_per_die;
    const double initiation_ns = transfer_time_ns(page_wire_bytes(), raw_bandwidth_GBps);

    // A codeword consumes only an initiation slot. Its response latency may
    // overlap later codewords in the same per-die pipeline. Decode and encode
    // conservatively share this issue port, so mixed traffic still contends.
    const auto issue = reserve(earliest_ns, initiation_ns, die.ecc_issue);
    const double finish_ns = issue.start_ns + latency_ns;
    if (!std::isfinite(finish_ns)) {
        throw std::runtime_error("HBF ECC completion time overflowed");
    }

    breakdown.ecc_queue_wait_ns += issue.wait_ns;
    breakdown.ecc_latency_ns += latency_ns;
    stats_.ecc_issue_busy_ns += initiation_ns;
    stats_.ecc_codeword_bytes = checked_add(
        stats_.ecc_codeword_bytes, page_wire_bytes(), "HBF ECC codeword bytes");
    die.ecc_issue_busy_ns += initiation_ns;
    die.ecc_inflight_intervals.push_back(DieState::EccInflightInterval{
        .start_ns = issue.start_ns,
        .finish_ns = finish_ns,
    });
    if (decode) {
        stats_.ecc_decode_ops++;
        stats_.ecc_decode_queue_wait_ns += issue.wait_ns;
        stats_.ecc_decode_latency_work_ns += latency_ns;
        stats_.ecc_decode_issue_busy_ns += initiation_ns;
        stats_.ecc_decode_codeword_bytes = checked_add(
            stats_.ecc_decode_codeword_bytes,
            page_wire_bytes(),
            "HBF ECC decode codeword bytes");
        die.ecc_decode_ops++;
    } else {
        stats_.ecc_encode_ops++;
        stats_.ecc_encode_queue_wait_ns += issue.wait_ns;
        stats_.ecc_encode_latency_work_ns += latency_ns;
        stats_.ecc_encode_issue_busy_ns += initiation_ns;
        stats_.ecc_encode_codeword_bytes = checked_add(
            stats_.ecc_encode_codeword_bytes,
            page_wire_bytes(),
            "HBF ECC encode codeword bytes");
        die.ecc_encode_ops++;
    }

    if (spans != nullptr) {
        const auto source_label = source_name(source);
        const auto operation =
            decode ? std::string{"decode"} : std::string{"encode"};
        const auto die_label = die_entity(addr);
        const auto issue_entity = die_label + "/ecc_issue_port";
        const auto pipeline_entity =
            die_label + "/ecc_" + operation + "_pipeline";
        trace_wait(
            spans,
            issue_entity,
            earliest_ns,
            issue.start_ns,
            "wait_ecc_" + operation + "_issue");
        const auto detail = std::to_string(page_wire_bytes()) +
            "B raw, II=" + fixed(initiation_ns, 6) + "ns";
        add_trace_span(
            spans,
            source_label + "/ecc_" + operation + "_issue",
            "ecc_issue",
            issue_entity,
            issue.start_ns,
            issue.finish_ns,
            false,
            detail);
        add_trace_span(
            spans,
            source_label + "/ecc_" + operation + "_latency",
            "ecc_latency",
            pipeline_entity,
            issue.start_ns,
            finish_ns,
            true,
            detail);
    }
    return finish_ns;
}

std::vector<HbfController::PageRunSpan>
HbfController::static_page_run_spans(
    std::uint64_t first_source_page,
    std::uint64_t pages) const {
    if (pages == 0) {
        throw std::runtime_error("HBF static page run cannot be empty");
    }
    const auto end_source_page = checked_add(
        first_source_page, pages, "HBF static page-run source end");
    if (end_source_page > total_pages_) {
        throw std::runtime_error(
            "HBF static page run exceeds physical page capacity");
    }
    const auto plane_count = static_cast<std::uint64_t>(planes_.size());
    std::vector<PageRunSpan> spans;
    std::vector<std::optional<std::size_t>> last_span_by_plane(
        planes_.size());
    const auto append = [&](std::size_t plane,
                            std::uint64_t first_page,
                            std::uint64_t count) {
        if (count == 0) {
            return;
        }
        auto& last = last_span_by_plane.at(plane);
        if (last &&
            spans.at(*last).first_page + spans.at(*last).page_count ==
                first_page) {
            spans.at(*last).page_count = checked_add(
                spans.at(*last).page_count,
                count,
                "HBF static page-run merged span");
            return;
        }
        last = spans.size();
        spans.push_back(PageRunSpan{
            .plane = plane,
            .first_page = first_page,
            .page_count = count,
        });
    };
    const auto append_source_page = [&](std::uint64_t source_page) {
        const auto ppn = static_ppn_for_source_page(source_page);
        const auto pages_per_physical_plane = checked_mul(
            config_.device.blocks_per_plane,
            config_.device.pages_per_block,
            "HBF static pages per plane");
        const auto plane = static_cast<std::size_t>(
            ppn / pages_per_physical_plane);
        append(plane, ppn % pages_per_physical_plane, 1);
    };

    auto cursor = first_source_page;
    if (cursor % plane_count != 0) {
        const auto next_cycle = checked_mul(
            checked_add(
                cursor / plane_count,
                1,
                "HBF static leading cycle"),
            plane_count,
            "HBF static leading cycle boundary");
        const auto leading_end = std::min(end_source_page, next_cycle);
        while (cursor < leading_end) {
            append_source_page(cursor++);
        }
    }
    if (cursor < end_source_page) {
        const auto full_cycle_end =
            end_source_page / plane_count * plane_count;
        if (cursor < full_cycle_end) {
            const auto first_page_in_plane = cursor / plane_count;
            const auto cycles = (full_cycle_end - cursor) / plane_count;
            for (std::size_t plane = 0; plane < planes_.size(); ++plane) {
                append(plane, first_page_in_plane, cycles);
            }
            cursor = full_cycle_end;
        }
        while (cursor < end_source_page) {
            append_source_page(cursor++);
        }
    }
    std::uint64_t represented_pages = 0;
    for (const auto& span : spans) {
        represented_pages = checked_add(
            represented_pages,
            span.page_count,
            "HBF static page-run represented pages");
    }
    if (represented_pages != pages) {
        throw std::runtime_error(
            "HBF static page-run span accounting did not conserve pages");
    }
    return spans;
}

std::uint64_t HbfController::static_ppn_for_source_page(
    std::uint64_t source_page) const {
    if (source_page >= total_pages_) {
        throw std::runtime_error(
            "HBF static source page exceeds physical page capacity");
    }
    const auto stack = stack_for_lpn(source_page);
    const auto stack_page_index = source_page / config_.device.stacks;
    const auto local_plane = stack_page_index % planes_per_stack();
    const auto page_in_plane = stack_page_index / planes_per_stack();
    const auto plane = stack * planes_per_stack() + local_plane;
    const auto pages_per_physical_plane = checked_mul(
        config_.device.blocks_per_plane,
        config_.device.pages_per_block,
        "HBF static pages per plane");
    return checked_add(
        checked_mul(
            plane,
            pages_per_physical_plane,
            "HBF static PPN plane base"),
        page_in_plane,
        "HBF static PPN");
}


void HbfController::record_mutated_lpn_range(
    std::uint64_t first_lpn,
    std::uint64_t pages) {
    if (pages == 0) {
        throw std::runtime_error("HBF mutation range cannot be empty");
    }
    auto begin = first_lpn;
    auto end = checked_add(first_lpn, pages, "HBF mutation range end");
    auto next = mutated_lpn_ranges_.lower_bound(begin);
    if (next != mutated_lpn_ranges_.begin()) {
        auto previous = std::prev(next);
        if (previous->second >= begin) {
            begin = previous->first;
            end = std::max(end, previous->second);
            next = mutated_lpn_ranges_.erase(previous);
        }
    }
    while (next != mutated_lpn_ranges_.end() && next->first <= end) {
        end = std::max(end, next->second);
        next = mutated_lpn_ranges_.erase(next);
    }
    const auto inserted = mutated_lpn_ranges_.emplace(begin, end).second;
    if (!inserted) {
        throw std::runtime_error("HBF mutation range insertion collided");
    }
}

bool HbfController::mutated_lpn_range_overlaps(
    std::uint64_t first_lpn,
    std::uint64_t pages) const {
    if (pages == 0) {
        throw std::runtime_error("HBF mutation overlap range cannot be empty");
    }
    const auto end = checked_add(
        first_lpn, pages, "HBF mutation overlap range end");
    auto next = mutated_lpn_ranges_.lower_bound(first_lpn);
    if (next != mutated_lpn_ranges_.begin()) {
        const auto previous = std::prev(next);
        if (previous->second > first_lpn) {
            return true;
        }
    }
    return next != mutated_lpn_ranges_.end() && next->first < end;
}



void HbfController::thermal_advance(
    ThermalNodeState& node,
    double t_ns,
    bool record) const {
    const double target_ns = std::max(t_ns, node.clock_ns);
    const double idle_c = thermal_idle_temperature_c_;
    const auto decay_to = [&](double until_ns) {
        double dt_ns = until_ns - node.clock_ns;
        if (dt_ns <= 0.0) {
            return;
        }
        if (node.throttled) {
            // Only deposits raise temperature, so a decay segment can only
            // cool: solve the exact release-threshold crossing instead of
            // sampling it.
            if (node.temperature_c > config_.device.thermal_release_c) {
                const double gap = node.temperature_c - idle_c;
                const double release_gap =
                    config_.device.thermal_release_c - idle_c;
                const double crossing_ns =
                    thermal_tau_ns_ * std::log(gap / release_gap);
                if (crossing_ns >= dt_ns) {
                    if (record) {
                        stats_.thermal_throttled_span_ns += dt_ns;
                    }
                    node.temperature_c = idle_c +
                        gap * std::exp(-dt_ns / thermal_tau_ns_);
                    node.clock_ns = until_ns;
                    return;
                }
                if (record) {
                    stats_.thermal_throttled_span_ns += crossing_ns;
                }
                node.temperature_c = config_.device.thermal_release_c;
                node.clock_ns += crossing_ns;
                dt_ns -= crossing_ns;
            }
            node.throttled = false;
        }
        node.temperature_c = idle_c +
            (node.temperature_c - idle_c) * std::exp(-dt_ns / thermal_tau_ns_);
        node.clock_ns = until_ns;
    };
    decay_to(target_ns);
}

HbfController::ThermalAdmission HbfController::thermal_pace_media(
    std::size_t stack,
    double earliest_ns,
    double energy_j,
    std::uint64_t media_ops,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    if (!config_.device.thermal_enabled || energy_j <= 0.0) {
        return ThermalAdmission{earliest_ns, earliest_ns};
    }
    auto& state = thermal_stacks_.at(stack);
    // The sensor is advanced to the admission instant before this work's
    // own heat is considered: firmware reacts to already dissipated work,
    // never to the command it is about to issue.
    thermal_advance(state.node, earliest_ns, true);
    if (!state.node.throttled) {
        return ThermalAdmission{earliest_ns, earliest_ns};
    }
    const double slot_ns = energy_j / thermal_pacing_power_w_ * 1e9;
    const auto budget = reserve(earliest_ns, slot_ns, state.pacing);
    const double start_ns = budget.start_ns;
    const double wait_ns = budget.wait_ns;
    breakdown.scheduler_queue_wait_ns += wait_ns;
    if (spans != nullptr) {
        trace_wait(
            spans,
            "stack" + std::to_string(stack) + "/thermal",
            earliest_ns,
            start_ns,
            "wait_thermal_pacing");
        add_trace_span(
            spans, "thermal_pacing_budget", "thermal_budget",
            "stack" + std::to_string(stack) + "/thermal",
            budget.start_ns, budget.finish_ns, false);
    }
    stats_.thermal_throttled_media_ops = checked_add(
        stats_.thermal_throttled_media_ops,
        media_ops,
        "HBF thermal paced media ops");
    stats_.thermal_throttle_wait_ns += wait_ns;
    stats_.thermal_pacing_busy_ns += slot_ns;
    // Return this operation's budget end, not the calendar's furthest
    // reservation: an admission can fit before already scheduled programs.
    return ThermalAdmission{start_ns, budget.finish_ns};
}

void HbfController::thermal_deposit_media(
    std::size_t stack,
    double at_ns,
    double energy_j) {
    if (!config_.device.thermal_enabled || energy_j <= 0.0) {
        return;
    }
    auto& state = thermal_stacks_.at(stack);
    thermal_advance(state.node, at_ns, true);
    stats_.thermal_media_energy_j += energy_j;
    state.node.temperature_c +=
        energy_j / config_.device.thermal_capacitance_j_per_c;
    state.node.peak_c =
        std::max(state.node.peak_c, state.node.temperature_c);
    if (!state.node.throttled &&
        state.node.temperature_c >= config_.device.thermal_throttle_c) {
        state.node.throttled = true;
        stats_.thermal_throttle_engagements = checked_add(
            stats_.thermal_throttle_engagements,
            1,
            "HBF thermal throttle engagements");
    }
}

HbfController::ThermalAdmission HbfController::thermal_admit_media(
    std::size_t stack,
    double earliest_ns,
    double energy_j,
    std::uint64_t media_ops,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto admission = thermal_pace_media(
        stack, earliest_ns, energy_j, media_ops, breakdown, spans);
    thermal_deposit_media(stack, admission.ready_ns, energy_j);
    return admission;
}

HbfController::ReadPipelineGroup HbfController::make_read_pipeline_group() const {
    const auto& d = config_.device;
    return {
        transfer_time_ns(d.command_address_bytes, d.tsv_bandwidth_GBps),
        transfer_time_ns(d.command_address_bytes, d.channel_bandwidth_GBps),
        transfer_time_ns(page_wire_bytes(), d.media_lane_bandwidth_GBps),
        transfer_time_ns(page_wire_bytes(), d.page_buffer_bandwidth_GBps),
        transfer_time_ns(page_wire_bytes(), d.channel_bandwidth_GBps),
        transfer_time_ns(page_wire_bytes(), d.tsv_bandwidth_GBps),
        transfer_time_ns(page_wire_bytes(), d.ecc_decode_raw_bandwidth_GBps_per_die),
        transfer_time_ns(d.page_size_bytes, d.logic_sram_bandwidth_GBps),
    };
}

double HbfController::schedule_read_group_page(
    std::uint64_t ppn, double earliest_ns, std::uint64_t payload_bytes,
    Breakdown& b, double& decoded_ready_ns, ReadPipelineGroup& group) {
    // Numeric foreground-read kernel. The caller has excluded tracing,
    // overlapping program/erase windows. Keeping the
    // page-major order is essential: commands and data share the TSV, and
    // later commands must still be allowed to occupy its earlier idle gaps.
    // The general scheduler below handles the excluded eventful cases.
    const bool frontier_changed =
        group.prepared_watermark_ns != reservation_causal_watermark_ns_;
    if (group.pages == 0 || frontier_changed) {
        ++stats_.read_pipeline_groups;
    }
    if (frontier_changed) {
        // Consecutive groups share the same causal frontier. Prepare shared
        // calendars only when it advances, not by scanning every channel/die
        // again at each group boundary. A mapping/GC dependency that advances
        // the frontier invalidates this preparation before the next page.
        group.prepared_watermark_ns = reservation_causal_watermark_ns_;
        for (auto& logic : logic_dies_) {
            logic.tsv.prune_before(reservation_causal_watermark_ns_);
            logic.sram.prune_before(reservation_causal_watermark_ns_);
        }
        for (auto& channel : channels_) {
            channel.command.prune_before(reservation_causal_watermark_ns_);
            channel.data.prune_before(reservation_causal_watermark_ns_);
        }
        for (auto& die : dies_) {
            die.source_queues[source_index(TransactionSource::User)].prune_before(reservation_causal_watermark_ns_);
            die.sequencer.prune_before(reservation_causal_watermark_ns_);
            die.ecc_issue.prune_before(reservation_causal_watermark_ns_);
        }
    }
    ++group.pages;
    ++stats_.read_pipeline_group_pages;
    const auto& d = config_.device;
    const auto addr = decode_ppn(ppn);
    auto& logic = logic_dies_[stack_index(addr)];
    auto& channel = channels_[channel_index(addr)];
    auto& die = dies_[die_index(addr)];
    auto& plane = planes_[plane_index(addr)];
    auto& sense_bank = plane.subarrays[subarray_index(addr)];
    auto& lane = plane.media_lanes[media_lane_index(addr)];
    auto& buffer = plane.page_buffer_banks[page_buffer_bank_index(addr)];
    // Only touched banks need preparation; shared paths were prepared above.
    sense_bank.timeline.prune_before(reservation_causal_watermark_ns_);
    lane.timeline.prune_before(reservation_causal_watermark_ns_);
    buffer.timeline.prune_before(reservation_causal_watermark_ns_);
    const auto slot = [](double at, double duration, ResourceTimeline& timeline) {
        const auto reserved = timeline.reserve(at, duration);
        return ScheduledTransfer{reserved.start_ns, reserved.finish_ns,
            std::max(0.0, reserved.start_ns - at)};
    };
    // The shared TSV and SRAM stages use ordered reservation streams. Commands
    // and data use independent certificates on the shared TSV; their original
    // page-major insertion order and every idle interval remain unchanged.
    // Time reversals, a changed stage duration or an invalidated calendar are
    // handled by reserve_ordered's exact first-fit fallback.
    const auto ordered_slot = [this](double at, double duration,
        ResourceTimeline& timeline, unsigned stream = 0) {
        const auto before = timeline.ordered_position_reuses();
        const auto reserved = timeline.reserve_ordered(at, duration, stream);
        stats_.read_pipeline_direct_gap_reservations +=
            timeline.ordered_position_reuses() - before;
        return ScheduledTransfer{reserved.start_ns, reserved.finish_ns,
            std::max(0.0, reserved.start_ns - at)};
    };
    earliest_ns = std::max(earliest_ns, media_->read_ready(ppn));
    thermal_deposit_media(stack_index(addr), earliest_ns, thermal_read_energy_j_);
    const auto command_received = schedule_external_request_command(channel_index(addr),
        earliest_ns, b, nullptr, "OCP page command", "user");
    b.tsv_transfer_ns += group.command_tsv_ns;
    const auto command_tsv = ordered_slot(command_received, group.command_tsv_ns, logic.tsv);
    b.scheduler_queue_wait_ns += command_tsv.wait_ns;
    const auto command = slot(command_tsv.finish_ns, group.command_channel_ns, channel.command);
    b.scheduler_queue_wait_ns += command.wait_ns;
    b.channel_transfer_ns += group.command_channel_ns;
    channel.command_busy_ns += group.command_channel_ns;
    ++channel.command_count;

    ++stats_.flash_scheduler_enqueues;
    const auto source = slot(command.finish_ns, d.flash_tsu_issue_ns,
        die.source_queues[source_index(TransactionSource::User)]);
    b.scheduler_queue_wait_ns += source.wait_ns;
    const auto issue = slot(source.start_ns, d.flash_tsu_issue_ns, die.sequencer);
    b.scheduler_queue_wait_ns += issue.wait_ns;
    b.command_ns += d.flash_tsu_issue_ns;
    die.sequencer_busy_ns += d.flash_tsu_issue_ns;
    ++die.transaction_count;
    ++stats_.flash_scheduler_issues;

    const auto sense = slot(issue.finish_ns, d.t_read_page_ns, sense_bank.timeline);
    const auto transfer = slot(sense.finish_ns, group.lane_ns, lane.timeline);
    const auto staged = slot(transfer.finish_ns, group.buffer_ns, buffer.timeline);
    b.scheduler_queue_wait_ns += sense.wait_ns;
    b.array_read_ns += d.t_read_page_ns;
    sense_bank.busy_ns += d.t_read_page_ns;
    ++sense_bank.read_count;
    b.scheduler_queue_wait_ns += transfer.wait_ns;
    b.media_lane_transfer_ns += group.lane_ns;
    lane.busy_ns += group.lane_ns;
    ++lane.read_count;
    record_plane_media_busy(plane, sense.start_ns, sense.start_ns + d.t_read_page_ns);
    ++plane.read_count;
    b.scheduler_queue_wait_ns += staged.wait_ns;
    b.page_buffer_ns += group.buffer_ns;
    buffer.busy_ns += group.buffer_ns;
    ++buffer.read_count;
    auto& block = blocks_[ppn / d.pages_per_block];
    block.issued_media_ready_ns = std::max(block.issued_media_ready_ns, staged.finish_ns);

    const auto data = slot(staged.finish_ns, group.channel_ns, channel.data);
    b.scheduler_queue_wait_ns += data.wait_ns;
    b.channel_transfer_ns += group.channel_ns;
    channel.data_busy_ns += group.channel_ns;
    ++channel.data_count;
    b.tsv_transfer_ns += group.tsv_ns;
    const auto tsv = ordered_slot(data.finish_ns, group.tsv_ns, logic.tsv, 1);
    b.scheduler_queue_wait_ns += tsv.wait_ns;
    const auto ecc = slot(tsv.finish_ns, group.ecc_issue_ns, die.ecc_issue);
    const double ecc_done = ecc.start_ns + d.ecc_decode_latency_ns;
    if (!std::isfinite(ecc_done)) throw std::runtime_error("HBF ECC completion time overflowed");
    b.ecc_queue_wait_ns += ecc.wait_ns;
    b.ecc_latency_ns += d.ecc_decode_latency_ns;
    stats_.ecc_issue_busy_ns += group.ecc_issue_ns;
    stats_.ecc_codeword_bytes = checked_add(stats_.ecc_codeword_bytes,
        page_wire_bytes(), "HBF ECC codeword bytes");
    die.ecc_issue_busy_ns += group.ecc_issue_ns;
    die.ecc_inflight_intervals.push_back({ecc.start_ns, ecc_done});
    ++stats_.ecc_decode_ops;
    stats_.ecc_decode_queue_wait_ns += ecc.wait_ns;
    stats_.ecc_decode_latency_work_ns += d.ecc_decode_latency_ns;
    stats_.ecc_decode_issue_busy_ns += group.ecc_issue_ns;
    stats_.ecc_decode_codeword_bytes = checked_add(stats_.ecc_decode_codeword_bytes,
        page_wire_bytes(), "HBF ECC decode codeword bytes");
    ++die.ecc_decode_ops;
    const auto sram = ordered_slot(ecc_done, group.sram_ns, logic.sram);
    b.scheduler_queue_wait_ns += sram.wait_ns;
    b.sram_staging_ns += group.sram_ns;
    decoded_ready_ns = sram.finish_ns;
    return schedule_external_read_egress(channel_index(addr), payload_bytes,
        sram.finish_ns, b, nullptr, "ocp_read_response", "decoded payload");
}

double HbfController::schedule_read_page(
    std::uint64_t ppn,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    HeatmapTrafficSource heatmap_attribution,
    ReadPayloadRoute route,
    std::uint64_t external_payload_bytes,
    double* decoded_ready_ns,
    bool thermally_preadmitted) {
    earliest_ns = std::max(earliest_ns, media_->read_ready(ppn));
    const auto addr = decode_ppn(ppn);
    auto& logic_die = logic_dies_.at(stack_index(addr));
    auto& channel = channels_.at(channel_index(addr));
    auto& plane = planes_.at(plane_index(addr));
    std::string source_label;
    if (spans != nullptr) {
        source_label = source_name(source);
    }
    const auto subarray_index_value = subarray_index(addr);
    const auto lane_index = media_lane_index(addr);
    const auto buffer_bank_index = page_buffer_bank_index(addr);
    auto& subarray = plane.subarrays.at(subarray_index_value);
    auto& lane = plane.media_lanes.at(lane_index);
    auto& buffer_bank = plane.page_buffer_banks.at(buffer_bank_index);
    if (thermally_preadmitted) {
        // The enclosing scalar request already paced this stack; the page
        // contributes its exact media heat here.
        thermal_deposit_media(
            stack_index(addr), earliest_ns, thermal_read_energy_j_);
    } else {
        earliest_ns = thermal_admit_media(
            stack_index(addr),
            earliest_ns,
            thermal_read_energy_j_,
            1,
            breakdown,
            spans).ready_ns;
    }
    const bool local_metadata = source == TransactionSource::Mapping && mapping_is_local();
    const bool logical_command_received = mapping_is_local() && source == TransactionSource::User;
    const double command_done = schedule_command_path(
        addr, earliest_ns, breakdown, spans, source, !local_metadata && !logical_command_received);
    auto tsu = schedule_flash_transaction(
        addr,
        command_done,
        breakdown,
        spans,
        source,
        TransactionKind::Read);

    const double lane_ns = transfer_time_ns(
        page_wire_bytes(), config_.device.media_lane_bandwidth_GBps);
    const double page_buffer_ns = transfer_time_ns(
        page_wire_bytes(), config_.device.page_buffer_bandwidth_GBps);
    ScheduledTransfer sense;
    ScheduledTransfer lane_transfer;
    ScheduledTransfer page_buffer;
    double local_ready_ns = tsu.finish_ns;
    for (;;) {
        subarray.timeline.prune_before(reservation_causal_watermark_ns_);
        lane.timeline.prune_before(reservation_causal_watermark_ns_);
        buffer_bank.timeline.prune_before(reservation_causal_watermark_ns_);

        const double sense_start_ns = subarray.timeline.preview_start(
            local_ready_ns, config_.device.t_read_page_ns);
        const double sense_finish_ns = sense_start_ns + config_.device.t_read_page_ns;
        const double lane_start_ns = lane.timeline.preview_start(
            sense_finish_ns, lane_ns);
        const double lane_finish_ns = lane_start_ns + lane_ns;
        const double buffer_start_ns = buffer_bank.timeline.preview_start(
            lane_finish_ns, page_buffer_ns);
        const double buffer_finish_ns = buffer_start_ns + page_buffer_ns;

        const auto conflict = std::lower_bound(
            plane.full_plane_windows.begin(),
            plane.full_plane_windows.end(),
            sense_start_ns,
            [](const PlaneState::BusyWindow& window, double start) {
                return window.end_ns <= start;
            });
        if (conflict != plane.full_plane_windows.end() &&
            conflict->begin_ns < buffer_finish_ns) {
            local_ready_ns = std::max(local_ready_ns, conflict->end_ns);
            continue;
        }

        sense = reserve(sense_start_ns, config_.device.t_read_page_ns, subarray.timeline);
        if (sense.start_ns != sense_start_ns)
            throw std::runtime_error("HBF bank sense moved after path preview");
        sense.wait_ns = std::max(0.0, sense.start_ns - tsu.finish_ns);
        lane_transfer = reserve(lane_start_ns, lane_ns, lane.timeline);
        page_buffer = reserve(
            buffer_start_ns, page_buffer_ns, buffer_bank.timeline);
        if (lane_transfer.start_ns != lane_start_ns ||
            page_buffer.start_ns != buffer_start_ns) {
            throw std::runtime_error(
                "HBF read local data path moved after atomic preview");
        }
        // Exact commits use the already-previewed start as their reservation
        // key, so reserve() correctly reports zero wait. Preserve the actual
        // pipeline queueing relative to the producer instead: otherwise the
        // trace shows lane/page-buffer stalls that disappear from Breakdown.
        lane_transfer.wait_ns = std::max(
            0.0, lane_transfer.start_ns - sense.finish_ns);
        page_buffer.wait_ns = std::max(
            0.0, page_buffer.start_ns - lane_transfer.finish_ns);
        break;
    }
    const double media_start = sense.start_ns;
    breakdown.scheduler_queue_wait_ns += std::max(0.0, media_start - tsu.finish_ns);
    const double sense_done = media_start + config_.device.t_read_page_ns;
    breakdown.array_read_ns += config_.device.t_read_page_ns;
    if (spans != nullptr) {
        const auto entity = subarray_entity(addr, subarray_index_value);
        trace_wait(
            spans, entity, tsu.finish_ns, media_start, "wait_subarray");
        add_trace_span(
            spans,
            source_label + "/array_read",
            "flash_array",
            entity,
            media_start,
            sense_done,
            true,
            "plane" + std::to_string(addr.plane) +
                " subarray" + std::to_string(subarray_index_value) +
                " lane" + std::to_string(lane_index));
    }
    subarray.busy_ns += config_.device.t_read_page_ns;
    subarray.read_count++;

    breakdown.scheduler_queue_wait_ns += lane_transfer.wait_ns;
    breakdown.media_lane_transfer_ns += lane_ns;
    if (spans != nullptr) {
        const auto entity = media_lane_entity(addr, lane_index);
        trace_wait(
            spans,
            entity,
            sense_done,
            lane_transfer.start_ns,
            "wait_media_lane");
        add_trace_span(
            spans,
            source_label + "/array_to_page_buffer",
            "media_lane",
            entity,
            lane_transfer.start_ns,
            lane_transfer.finish_ns,
            true,
            std::to_string(page_wire_bytes()) + "B raw from subarray" +
                std::to_string(subarray_index_value));
    }
    lane.busy_ns += lane_ns;
    lane.read_count++;
    record_plane_media_busy(plane, media_start, sense_done);
    plane.read_count++;

    breakdown.scheduler_queue_wait_ns += page_buffer.wait_ns;
    breakdown.page_buffer_ns += page_buffer_ns;
    buffer_bank.busy_ns += page_buffer_ns;
    buffer_bank.read_count++;
    if (spans != nullptr) {
        const auto entity = page_buffer_bank_entity(addr, buffer_bank_index);
        trace_wait(
            spans,
            entity,
            lane_transfer.finish_ns,
            page_buffer.start_ns,
            "wait_page_buffer_bank");
        add_trace_span(
            spans,
            source_label + "/page_buffer_out",
            "page_buffer",
            entity,
            page_buffer.start_ns,
            page_buffer.finish_ns,
            true,
            std::to_string(page_wire_bytes()) + "B raw bank" +
                std::to_string(buffer_bank_index));
    }
    auto& block = blocks_.at(static_cast<std::size_t>(
        ppn / config_.device.pages_per_block));
    block.issued_media_ready_ns = std::max(
        block.issued_media_ready_ns, page_buffer.finish_ns);

    const double channel_ns = transfer_time_ns(page_wire_bytes(), config_.device.channel_bandwidth_GBps);
    auto channel_transfer = reserve(page_buffer.finish_ns, channel_ns, channel.data);
    breakdown.scheduler_queue_wait_ns += channel_transfer.wait_ns;
    breakdown.channel_transfer_ns += channel_ns;
    channel.data_busy_ns += channel_ns;
    channel.data_count++;
    if (spans != nullptr) {
        const auto entity = channel_entity(addr);
        trace_wait(
            spans,
            entity,
            page_buffer.finish_ns,
            channel_transfer.start_ns,
            "wait_channel_data");
        add_trace_span(
            spans,
            source_label + "/data_out_channel",
            "flash_channel",
            entity,
            channel_transfer.start_ns,
            channel_transfer.finish_ns,
            true,
            std::to_string(page_wire_bytes()) + "B raw");
    }

    const double tsv_ns = transfer_time_ns(page_wire_bytes(), config_.device.tsv_bandwidth_GBps);
    breakdown.tsv_transfer_ns += tsv_ns;
    auto tsv = reserve(channel_transfer.finish_ns, tsv_ns, logic_die.tsv);
    breakdown.scheduler_queue_wait_ns += tsv.wait_ns;
    if (spans != nullptr) {
        const auto entity =
            "stack" + std::to_string(addr.stack) + "/tsv";
        trace_wait(
            spans,
            entity,
            channel_transfer.finish_ns,
            tsv.start_ns,
            "wait_tsv_data");
        add_trace_span(
            spans,
            source_label + "/data_out_tsv",
            "tsv",
            entity,
            tsv.start_ns,
            tsv.finish_ns,
            true,
            std::to_string(page_wire_bytes()) + "B raw");
    }

    // Raw codeword bytes cross the flash channel and TSV. Decode happens on
    // the logic die before decoded payload enters SRAM and leaves over the
    // external HBIO; parity/OOB bytes never consume external payload BW.
    const double ecc_done = schedule_ecc(
        addr, tsv.finish_ns, breakdown, spans, source, true);
    const double sram_ns = transfer_time_ns(
        config_.device.page_size_bytes, config_.device.logic_sram_bandwidth_GBps);
    auto sram = reserve(ecc_done, sram_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(addr.stack);
        trace_wait(spans, entity, ecc_done, sram.start_ns, "wait_sram");
        add_trace_span(
            spans,
            source_label + "/sram_stage_read",
            "sram",
            entity,
            sram.start_ns,
            sram.finish_ns,
            true,
            std::to_string(config_.device.page_size_bytes) + "B payload");
    }
    if (address_heatmap_ != nullptr) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::HbfPhysical,
            .direction = TrafficDirection::Read,
            .source = resolve_heatmap_source(source, heatmap_attribution),
            .address = checked_mul(
                ppn,
                config_.device.page_size_bytes,
                "HBF heatmap page-read address"),
            .bytes = config_.device.page_size_bytes,
        });
    }
    if (decoded_ready_ns != nullptr) {
        *decoded_ready_ns = sram.finish_ns;
    }
    if (route == ReadPayloadRoute::DeviceDram) return sram.finish_ns;
    if (route == ReadPayloadRoute::MappingBuffer && local_metadata)
        return mapping_memory_transfer(addr.stack, config_.device.page_size_bytes,
            Op::Write, sram.finish_ns, breakdown, spans);
    const auto returned_bytes = route != ReadPayloadRoute::External ?
        config_.device.page_size_bytes : external_payload_bytes;
    if (source == TransactionSource::Mapping) stats_.mapping_hbio_read_bytes += returned_bytes;
    const auto out_ns = schedule_external_read_egress(channel_index(addr), returned_bytes,
        sram.finish_ns, breakdown, spans, "ocp_read_response", "decoded payload");
    if (route == ReadPayloadRoute::MappingBuffer)
        return mapping_memory_transfer(addr.stack, returned_bytes, Op::Write, out_ns, breakdown, spans);
    return route == ReadPayloadRoute::HostBuffer ?
        host_memory_transfer(addr.stack, returned_bytes, Op::Write, out_ns, breakdown, spans) : out_ns;
}

double HbfController::schedule_program_page(
    std::uint64_t ppn,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    HeatmapTrafficSource heatmap_attribution,
    double* source_consumed_ns) {
    const auto addr = decode_ppn(ppn);
    seed_media_image();
    const bool auto_erase = media_->prepare_program(ppn);
    const bool local_metadata = source == TransactionSource::Mapping && mapping_is_local();
    const auto command_received = mapping_is_local() &&
        (source == TransactionSource::Mapping || source == TransactionSource::User) ? earliest_ns :
        schedule_external_request_command(channel_index(addr),
            earliest_ns, breakdown, spans, "complete-page write", source_name(source));
    auto erase_ready = command_received;
    if (auto_erase) {
        ++stats_.auto_erase_requests;
        erase_ready = schedule_erase_block(ppn / config_.device.pages_per_block,
            command_received, breakdown, spans, source, heatmap_attribution, true);
    }
    // Mapping checkpoints and GC copies source the reserved HBM buffer.
    // Foreground pages are supplied by the caller or an already-read write
    // buffer: charging another source read here would count that data twice.
    const auto source_ready = source == TransactionSource::Mapping ?
        mapping_memory_transfer(addr.stack, config_.device.page_size_bytes, Op::Read,
            earliest_ns, breakdown, spans) : source == TransactionSource::Host ?
        host_memory_transfer(addr.stack, config_.device.page_size_bytes, Op::Read,
            earliest_ns, breakdown, spans) : earliest_ns;
    const auto received = local_metadata ? source_ready :
        program_payload_on_device_ ? std::max(command_received, source_ready) :
        schedule_external_write_ingress(channel_index(addr),
            config_.device.page_size_bytes, std::max(command_received, source_ready), breakdown, spans,
            "complete host page", "ocp_page_ingress");
    if (source == TransactionSource::Mapping && !local_metadata)
        stats_.mapping_hbio_write_bytes += config_.device.page_size_bytes;
    return schedule_buffered_program_page(ppn, std::max(erase_ready, received),
        breakdown, spans, source, heatmap_attribution, source_consumed_ns);
}

double HbfController::schedule_buffered_program_page(
    std::uint64_t ppn, double earliest_ns, Breakdown& breakdown,
    std::vector<TraceSpan>* spans, TransactionSource source,
    HeatmapTrafficSource heatmap_attribution, double* source_consumed_ns) {
    const auto addr = decode_ppn(ppn);
    auto& logic_die = logic_dies_.at(stack_index(addr));
    auto& channel = channels_.at(channel_index(addr));
    auto& plane = planes_.at(plane_index(addr));
    std::string source_label;
    if (spans != nullptr) {
        source_label = source_name(source);
    }
    earliest_ns = thermal_admit_media(
        stack_index(addr),
        earliest_ns,
        thermal_program_energy_j_,
        1,
        breakdown,
        spans).ready_ns;
    const double command_done = schedule_command_path(
        addr, earliest_ns, breakdown, spans, source, false);

    // The complete page is now in device SRAM. Host metadata uses the same
    // channel interface as data; the device-local control sources private memory.
    // Programming reads one full decoded page into ECC, then sends the raw
    // page+OOB codeword over TSV and the flash channel.
    const double sram_ns = transfer_time_ns(
        config_.device.page_size_bytes, config_.device.logic_sram_bandwidth_GBps);
    auto sram = reserve(command_done, sram_ns, logic_die.sram);
    if (source_consumed_ns) *source_consumed_ns = sram.finish_ns;
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(addr.stack);
        trace_wait(
            spans, entity, command_done, sram.start_ns, "wait_sram");
        add_trace_span(
            spans,
            source_label + "/sram_stage_write",
            "sram",
            entity,
            sram.start_ns,
            sram.finish_ns,
            true,
            std::to_string(config_.device.page_size_bytes) + "B payload");
    }

    const double ecc_done = schedule_ecc(
        addr, sram.finish_ns, breakdown, spans, source, false);

    const double tsv_ns = transfer_time_ns(page_wire_bytes(), config_.device.tsv_bandwidth_GBps);
    breakdown.tsv_transfer_ns += tsv_ns;
    auto tsv = reserve(ecc_done, tsv_ns, logic_die.tsv);
    breakdown.scheduler_queue_wait_ns += tsv.wait_ns;
    if (spans != nullptr) {
        const auto entity =
            "stack" + std::to_string(addr.stack) + "/tsv";
        trace_wait(spans, entity, ecc_done, tsv.start_ns, "wait_tsv_data");
        add_trace_span(
            spans,
            source_label + "/data_in_tsv",
            "tsv",
            entity,
            tsv.start_ns,
            tsv.finish_ns,
            true,
            std::to_string(page_wire_bytes()) + "B raw");
    }

    const double channel_ns = transfer_time_ns(page_wire_bytes(), config_.device.channel_bandwidth_GBps);
    auto channel_transfer = reserve(tsv.finish_ns, channel_ns, channel.data);
    breakdown.scheduler_queue_wait_ns += channel_transfer.wait_ns;
    breakdown.channel_transfer_ns += channel_ns;
    channel.data_busy_ns += channel_ns;
    channel.data_count++;
    if (spans != nullptr) {
        const auto entity = channel_entity(addr);
        trace_wait(
            spans,
            entity,
            tsv.finish_ns,
            channel_transfer.start_ns,
            "wait_channel_data");
        add_trace_span(
            spans,
            source_label + "/data_in_channel",
            "flash_channel",
            entity,
            channel_transfer.start_ns,
            channel_transfer.finish_ns,
            true,
            std::to_string(page_wire_bytes()) + "B raw");
    }

    auto tsu = schedule_flash_transaction(
        addr,
        channel_transfer.finish_ns,
        breakdown,
        spans,
        source,
        TransactionKind::Program);

    const double barrier_ns = config_.device.t_program_page_ns;
    const double media_start = preview_full_plane_window(
        plane, tsu.finish_ns, barrier_ns);
    breakdown.scheduler_queue_wait_ns += std::max(0.0, media_start - tsu.finish_ns);
    // Program completion and every reserved calendar use the same interval.
    const double program_done = media_start + barrier_ns;
    breakdown.array_program_ns += barrier_ns;
    if (spans != nullptr) {
        const auto entity = plane_entity(addr);
        trace_wait(
            spans, entity, tsu.finish_ns, media_start, "wait_plane_array");
        add_trace_span(
            spans,
            source_label + "/array_program",
            "flash_array",
            entity,
            media_start,
            program_done);
    }
    const auto round_barrier = reserve(
        media_start, barrier_ns, plane.array_barrier);
    if (round_barrier.start_ns != media_start) {
        throw std::runtime_error(
            "HBF program barrier diverged from bank array calendar");
    }
    for (auto& subarray : plane.subarrays) {
        const auto barrier = reserve(media_start, barrier_ns, subarray.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF program barrier diverged across subarray calendars");
        }
    }
    // Program is a non-preemptible full-plane operation. The previous
    // suspend path could move this frontier after returning a completion,
    // violating causality, so it has been removed until a preemptible event
    // primitive with revisable completion dependencies exists.
    for (auto& lane : plane.media_lanes) {
        const auto barrier = reserve(media_start, barrier_ns, lane.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF program barrier diverged across media-lane calendars");
        }
    }
    for (auto& bank : plane.page_buffer_banks) {
        const auto barrier = reserve(media_start, barrier_ns, bank.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF program barrier diverged across page-buffer calendars");
        }
    }
    record_full_plane_window(plane, media_start, program_done);
    record_plane_media_busy(plane, media_start, program_done);
    plane.program_count++;
    auto& block = blocks_.at(static_cast<std::size_t>(
        ppn / config_.device.pages_per_block));
    block.issued_media_ready_ns = std::max(
        block.issued_media_ready_ns, program_done);
    if (address_heatmap_ != nullptr) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::HbfPhysical,
            .direction = TrafficDirection::Write,
            .source = resolve_heatmap_source(source, heatmap_attribution),
            .address = checked_mul(
                ppn,
                config_.device.page_size_bytes,
                "HBF heatmap page-program address"),
            .bytes = config_.device.page_size_bytes,
        });
    }
    media_->complete_program(ppn, program_done);
    return program_done;
}

double HbfController::schedule_erase_block(
    std::size_t block,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    HeatmapTrafficSource heatmap_attribution, bool auto_erase) {
    if (!auto_erase) media_->erase(block);
    const auto ppn = static_cast<std::uint64_t>(block) * config_.device.pages_per_block;
    const auto addr = decode_ppn(ppn);
    auto& plane = planes_.at(plane_index(addr));
    std::string source_label;
    if (spans != nullptr) {
        source_label = source_name(source);
    }

    if (blocks_.at(block).issued_media_ready_ns > earliest_ns) {
        if (spans != nullptr) {
            trace_wait(
                spans,
                addr.path(),
                earliest_ns,
                blocks_.at(block).issued_media_ready_ns,
                "wait_target_block_access");
        }
        breakdown.scheduler_queue_wait_ns +=
            blocks_.at(block).issued_media_ready_ns - earliest_ns;
        earliest_ns = blocks_.at(block).issued_media_ready_ns;
    }

    earliest_ns = thermal_admit_media(
        stack_index(addr),
        earliest_ns,
        thermal_erase_energy_j_,
        1,
        breakdown,
        spans).ready_ns;
    const double command_done = schedule_command_path(
        addr, earliest_ns, breakdown, spans, source, !auto_erase);
    auto tsu = schedule_flash_transaction(
        addr,
        command_done,
        breakdown,
        spans,
        source,
        TransactionKind::Erase);

    const double barrier_ns = config_.device.t_erase_block_ns;
    const double media_start = preview_full_plane_window(
        plane, tsu.finish_ns, barrier_ns);
    breakdown.scheduler_queue_wait_ns += std::max(0.0, media_start - tsu.finish_ns);
    const double erase_done = media_start + config_.device.t_erase_block_ns;
    breakdown.array_erase_ns += config_.device.t_erase_block_ns;
    if (spans != nullptr) {
        const auto entity = plane_entity(addr);
        trace_wait(
            spans, entity, tsu.finish_ns, media_start, "wait_plane_array");
        add_trace_span(
            spans,
            source_label + "/block_erase",
            "flash_array",
            entity,
            media_start,
            erase_done,
            true,
            "block" + std::to_string(addr.block));
    }
    const auto round_barrier = reserve(
        media_start, barrier_ns, plane.array_barrier);
    if (round_barrier.start_ns != media_start) {
        throw std::runtime_error(
            "HBF erase barrier diverged from bank array calendar");
    }
    for (auto& subarray : plane.subarrays) {
        const auto barrier = reserve(media_start, barrier_ns, subarray.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF erase barrier diverged across subarray calendars");
        }
    }
    for (auto& lane : plane.media_lanes) {
        const auto barrier = reserve(media_start, barrier_ns, lane.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF erase barrier diverged across media-lane calendars");
        }
    }
    for (auto& bank : plane.page_buffer_banks) {
        const auto barrier = reserve(media_start, barrier_ns, bank.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF erase barrier diverged across page-buffer calendars");
        }
    }
    record_full_plane_window(plane, media_start, erase_done);
    record_plane_media_busy(plane, media_start, erase_done);
    plane.erase_count++;
    stats_.block_erases++;
    // Wear checks follow actual physical erases, not administrative GC reclaim.
    if (auto_erase && config_.host.static_wear_leveling_erase_gap)
        pending_wear_check_by_stack_.at(stack_index(addr)) = block;
    // A scheduled erase is committed to happen and the block is already
    // unallocatable, so its P/E cycle is counted here, at issue, exactly like
    // stats_.block_erases; the later commit only resets the page state.
    auto& erased_block = blocks_.at(block);
    if (erased_block.erase_count == std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("HBF block erase count overflowed");
    }
    remove_managed_block_wear(block);
    erased_block.erase_count = media_->record_erase(block);
    add_managed_block_wear(block);
    mark_gc_candidate_dirty(block);
    if (address_heatmap_ != nullptr) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::HbfPhysical,
            .direction = TrafficDirection::Erase,
            .source = resolve_heatmap_source(source, heatmap_attribution),
            .address = checked_mul(
                ppn,
                config_.device.page_size_bytes,
                "HBF heatmap block-erase address"),
            .bytes = checked_mul(
                config_.device.pages_per_block,
                config_.device.page_size_bytes,
                "HBF heatmap block-erase bytes"),
        });
    }
    return erase_done;
}

void HbfController::refresh_mapping_issue_stats() const {
    // Preserve both port order and writeback: later request-level += updates
    // must start from the same floating-point baseline as a full observation.
    double mapping_dram_issue_busy_ns = 0.0;
    double write_buffer_dram_issue_busy_ns = 0.0;
    for (const auto& memory_port : mapping_memory_issue_by_stack_)
        mapping_dram_issue_busy_ns += memory_port.reserved_work_ns;
    for (const auto& logic_die : logic_dies_) {
        write_buffer_dram_issue_busy_ns +=
            logic_die.write_buffer_dram_issue.reserved_work_ns;
    }
    stats_.mapping_dram_issue_busy_ns = mapping_dram_issue_busy_ns;
    double compute_work = 0;
    for (const auto& worker : mapping_compute_workers_) compute_work += worker.reserved_work_ns;
    if (std::abs(compute_work - stats_.mapping_compute_work_ns) >
        std::max(1e-9, std::abs(compute_work) * 1e-12))
        throw std::runtime_error("mapping compute work accounting diverged");
    stats_.write_buffer_dram_issue_busy_ns =
        write_buffer_dram_issue_busy_ns;
}

void HbfController::refresh_accounting_stats() const {
    std::uint64_t free_pages = 0;
    std::uint64_t valid_pages = 0;
    std::uint64_t invalid_pages = 0;
    std::uint64_t pending_program_pages = 0;
    std::uint64_t pending_mapping_publications = 0;
    std::uint64_t static_unmaterialized_pages = 0;
    std::uint64_t raw_reserved_pages = 0;
    std::uint64_t writable_blocks = 0;
    std::uint64_t worn_blocks = 0;
    std::uint64_t block_erase_count_sum = 0;
    long double block_erase_count_sum_squares = 0.0L;
    std::uint32_t min_block_erase_count =
        std::numeric_limits<std::uint32_t>::max();
    std::map<std::uint32_t, std::uint64_t> block_erase_count_histogram;
    std::uint32_t max_block_erase_count = 0;
    std::vector<std::uint64_t> free_pages_per_stack(config_.device.stacks, 0);
    std::vector<std::uint8_t> free_pool_membership(blocks_.size(), 0);

    refresh_mapping_issue_stats();
    const auto mapping_table_bytes = checked_mul(
        mapping_table_bytes_per_stack_,
        config_.device.stacks,
        "HBF mapping-table accounting");
    if (stats_.mapping_table_bytes != mapping_table_bytes ||
        stats_.mapping_table_bytes_per_stack !=
            mapping_table_bytes_per_stack_ ||
        stats_.mapping_table_pages_per_stack !=
            mapping_table_pages_per_stack_ ||
        stats_.mapping_dram_resources != config_.device.stacks) {
        throw std::runtime_error(
            "HBF mapping-table footprint accounting diverged");
    }
    const auto expected_write_buffer_dram =
        derive_write_buffer_dram_capacity(config_);
    if (stats_.controller_dram_budget_bytes != config_.host.ctrl_dram_bytes ||
        stats_.controller_dram_budget_bytes_per_stack !=
            config_.host.ctrl_dram_bytes / config_.device.stacks ||
        stats_.write_buffer_capacity_bytes !=
            expected_write_buffer_dram.total_bytes ||
        stats_.write_buffer_capacity_bytes_per_stack !=
            expected_write_buffer_dram.bytes_per_stack) {
        throw std::runtime_error(
            "HBF shared controller-DRAM capacity accounting diverged");
    }
    if (config_.host.mapping_mode == MappingMode::FullResident) {
        const auto required_ctrl_dram_per_stack = checked_add(
            mapping_table_bytes_per_stack_ + config_.host.mapping_scratch_pages *
                (config_.device.page_size_bytes + config_.host.mapping_cache_tag_bytes),
            expected_write_buffer_dram.bytes_per_stack + stats_.host_gc_buffer_bytes / config_.device.stacks,
            "HBF resident controller-DRAM audit");
        if (stats_.resident_mapping_table_bytes != mapping_table_bytes ||
            stats_.resident_mapping_table_bytes_per_stack !=
                mapping_table_bytes_per_stack_ ||
            stats_.resident_mapping_pages_per_stack !=
                mapping_table_pages_per_stack_ ||
            config_.host.ctrl_dram_bytes / config_.device.stacks <
                required_ctrl_dram_per_stack ||
            stats_.mapping_directory_entry_bytes != 0 ||
            stats_.mapping_directory_bytes != 0 ||
            stats_.mapping_directory_bytes_per_stack != 0 ||
            stats_.mapping_cache_capacity_bytes != 0 ||
            stats_.mapping_cache_capacity_bytes_per_stack != 0 ||
            stats_.mapping_cache_pages_per_stack != 0 ||
            stats_.mapping_cache_entries != 0 ||
            stats_.mapping_cache_peak_entries != 0 ||
            stats_.mapping_cache_hits != 0 ||
            stats_.mapping_cache_misses != 0 ||
            stats_.mapping_cache_erased_misses != 0 ||
            stats_.mapping_cache_coalesced_misses != 0 ||
            stats_.mapping_cache_evictions != 0 ||
            stats_.mapping_cache_dirty_evictions != 0 ||
            stats_.mapping_media_reads != 0 ||
            stats_.mapping_media_read_bytes != 0) {
            throw std::runtime_error(
                "HBF full-resident mapping accounting diverged");
        }
    } else if (config_.host.mapping_mode == MappingMode::RawPhysical) {
        if (mapping_table_bytes != 0 ||
            stats_.mapping_table_bytes != 0 ||
            stats_.resident_mapping_table_bytes != 0 ||
            stats_.mapping_directory_bytes != 0 ||
            stats_.mapping_cache_capacity_bytes != 0 ||
            stats_.mapping_cache_entries != 0 ||
            stats_.mapping_cache_hits != 0 ||
            stats_.mapping_cache_misses != 0 ||
            stats_.mapping_media_reads != 0 ||
            stats_.mapping_lookup_ops != 0 ||
            stats_.mapping_update_ops != 0 ||
            stats_.mapping_dram_issue_busy_ns != 0.0 ||
            stats_.initial_mapping_pages != 0 ||
            !mapping_vpn_to_ppn_.empty()) {
            throw std::runtime_error(
                "HBF raw-physical mode accounting diverged: the exposed "
                "address space must carry no L2P state or work");
        }
    } else {
        std::uint64_t cache_entries = 0;
        for (std::size_t stack = 0; stack < config_.device.stacks; ++stack) {
            const auto& cache = mapping_cache_by_stack_.at(stack);
            const auto& lru = mapping_cache_lru_by_stack_.at(stack);
            if (cache.size() != lru.size() ||
                mapping_cache_bytes_by_stack_.at(stack) > stats_.mapping_cache_capacity_bytes_per_stack) {
                throw std::runtime_error(
                    "HBF mapping-cache state exceeds its stack partition");
            }
            std::uint64_t observed_bytes = 0;
            for (const auto& [key, entry] : cache) {
                (void)key;
                observed_bytes += entry.charged_bytes;
            }
            if (observed_bytes != mapping_cache_bytes_by_stack_.at(stack))
                throw std::runtime_error("HBF mapping cache byte ledger diverged");
            for (auto position = lru.begin(); position != lru.end();
                 ++position) {
                const auto found = cache.find(*position);
                if (found == cache.end() || found->second.iterator != position) {
                    throw std::runtime_error(
                        "HBF mapping-cache LRU index diverged");
                }
            }
            cache_entries = checked_add(
                cache_entries,
                cache.size(),
                "HBF mapping-cache entry accounting");
        }
        const auto cache_bytes_per_stack = checked_mul(
            mapping_cache_pages_per_stack_,
            config_.device.page_size_bytes + config_.host.mapping_cache_tag_bytes,
            "HBF mapping-cache bytes per stack audit");
        const auto cache_bytes = checked_mul(
            cache_bytes_per_stack,
            config_.device.stacks,
            "HBF mapping-cache total bytes audit");
        const auto directory_bytes_per_stack = checked_mul(
            mapping_table_pages_per_stack_,
            config_.host.mapping_directory_entry_bytes +
                (config_.host.mapping_cache_layout == MappingCacheLayout::Entry ? 8 : 0),
            "HBF mapping-directory bytes per stack audit");
        const auto directory_bytes = checked_mul(
            directory_bytes_per_stack,
            config_.device.stacks,
            "HBF mapping-directory total bytes audit");
        auto used_ctrl_dram_bytes_per_stack = checked_add(
            directory_bytes_per_stack,
            cache_bytes_per_stack,
            "HBF cached-mapping used DRAM per stack");
        used_ctrl_dram_bytes_per_stack = checked_add(
            used_ctrl_dram_bytes_per_stack,
            expected_write_buffer_dram.bytes_per_stack + stats_.host_gc_buffer_bytes / config_.device.stacks + config_.host.mapping_scratch_pages *
                (config_.device.page_size_bytes + config_.host.mapping_cache_tag_bytes),
            "HBF cached mapping plus write-buffer DRAM per stack");
        const auto mapping_accesses = checked_add(
            stats_.mapping_lookup_ops,
            stats_.mapping_update_ops,
            "HBF cached mapping accesses");
        if (stats_.resident_mapping_table_bytes != 0 ||
            stats_.resident_mapping_table_bytes_per_stack != 0 ||
            stats_.resident_mapping_pages_per_stack != 0 ||
            stats_.mapping_directory_entry_bytes !=
                config_.host.mapping_directory_entry_bytes +
                    (config_.host.mapping_cache_layout == MappingCacheLayout::Entry ? 8 : 0) ||
            stats_.mapping_directory_bytes != directory_bytes ||
            stats_.mapping_directory_bytes_per_stack !=
                directory_bytes_per_stack ||
            used_ctrl_dram_bytes_per_stack >
                config_.host.ctrl_dram_bytes / config_.device.stacks ||
            stats_.mapping_cache_capacity_bytes != cache_bytes ||
            stats_.mapping_cache_capacity_bytes_per_stack !=
                cache_bytes_per_stack ||
            stats_.mapping_cache_pages_per_stack !=
                mapping_cache_pages_per_stack_ ||
            stats_.mapping_cache_entries != cache_entries ||
            stats_.mapping_cache_peak_entries < cache_entries ||
            stats_.mapping_cache_peak_entries >
                checked_mul(
                    mapping_cache_record_limit_per_stack_,
                    config_.device.stacks,
                    "HBF mapping-cache peak capacity") ||
            checked_add(
                stats_.mapping_cache_hits,
                stats_.mapping_cache_misses,
                "HBF mapping-cache hit/miss accounting") !=
                mapping_accesses ||
            checked_add(
                checked_add(
                    stats_.mapping_media_reads + stats_.mapping_cache_buffer_hits,
                    stats_.mapping_cache_erased_misses,
                    "HBF mapping-cache direct miss classes"),
                stats_.mapping_cache_coalesced_misses,
                "HBF mapping-cache coalesced miss classes") !=
                stats_.mapping_cache_misses ||
            stats_.mapping_cache_dirty_evictions >
                stats_.mapping_cache_evictions ||
            stats_.mapping_cache_coalesced_misses >
                stats_.mapping_cache_misses ||
            stats_.mapping_buffer_coalesced_reads > stats_.mapping_cache_buffer_hits ||
            stats_.mapping_media_read_bytes != checked_mul(
                stats_.mapping_media_reads,
                config_.device.page_size_bytes,
                "HBF mapping-cache media-read accounting")) {
            throw std::runtime_error(
                "HBF cached mapping accounting diverged");
        }
    }
    if (stats_.mapping_lookup_ops != checked_add(
            stats_.mapping_user_lookup_ops,
            stats_.mapping_gc_lookup_ops,
            "HBF mapping lookup-source accounting") ||
        stats_.mapping_update_ops != checked_add(
            stats_.mapping_user_update_ops,
            stats_.mapping_gc_update_ops,
            "HBF mapping update-source accounting")) {
        throw std::runtime_error(
            "HBF mapping user/GC source accounting diverged");
    }
    const auto valid_mapping_wait = [](std::uint64_t operations,
                                       double total_ns,
                                       double max_ns) {
        if (!std::isfinite(total_ns) || !std::isfinite(max_ns) ||
            total_ns < 0.0 || max_ns < 0.0 || max_ns > total_ns) {
            return false;
        }
        return operations == 0 ?
            total_ns == 0.0 && max_ns == 0.0 :
            total_ns > 0.0 && max_ns > 0.0;
    };
    const auto sram_bypasses = stats_.mapping_sram_hits + stats_.mapping_sram_coalesced_misses;
    if (sram_bypasses > stats_.mapping_lookup_ops)
        throw std::runtime_error("mapping SRAM bypass count exceeds logical lookups");
    if (config_.host.mapping_sram_bytes) {
        std::uint64_t entries = 0;
        for (std::size_t stack = 0; stack < config_.device.stacks; ++stack) {
            const auto& cache = mapping_sram_by_stack_.at(stack);
            const auto& lru = mapping_sram_lru_by_stack_.at(stack);
            if (cache.size() != lru.size() || cache.size() >
                config_.host.mapping_sram_bytes / config_.device.stacks / mapping_sram_record_bytes)
                throw std::runtime_error("mapping SRAM exceeded its stack partition");
            entries += cache.size();
            for (const auto& [lpn, entry] : cache)
                if (entry.iterator == lru.end() || *entry.iterator != lpn ||
                    entry.fill_ready_ns > entry.evictable_ns)
                    throw std::runtime_error("mapping SRAM replacement state diverged");
        }
        if (entries != stats_.mapping_sram_entries ||
            entries + stats_.mapping_sram_evictions + stats_.mapping_sram_invalidations != stats_.mapping_sram_fills ||
            stats_.mapping_sram_hits + stats_.mapping_sram_misses != stats_.mapping_lookup_ops ||
            stats_.mapping_sram_fills + stats_.mapping_sram_coalesced_misses != stats_.mapping_sram_misses ||
            stats_.mapping_sram_read_bytes !=
                (stats_.mapping_lookup_ops + stats_.mapping_update_ops) * mapping_sram_record_bytes ||
            stats_.mapping_sram_write_bytes !=
                (stats_.mapping_sram_fills + stats_.mapping_sram_invalidations) * mapping_sram_record_bytes ||
            stats_.mapping_sram_peak_entries * mapping_sram_record_bytes > config_.host.mapping_sram_bytes)
            throw std::runtime_error("mapping SRAM traffic or capacity accounting diverged");
    }
    const double expected_mapping_issue_busy_ns =
        static_cast<double>(checked_add(
            checked_add(
                stats_.mapping_lookup_ops - sram_bypasses,
                stats_.mapping_update_ops,
                "HBF mapping access accounting"),
            stats_.mapping_directory_lookup_ops + stats_.mapping_buffer_lookup_ops +
                stats_.mapping_buffer_patch_ops + stats_.mapping_dirty_probe_ops +
                stats_.mapping_checkpoint_epoch_ops,
            "HBF mapping directory access accounting")) *
        config_.host.ctrl_dram_issue_ns;
    const double mapping_busy_tolerance = std::max(
        1e-9,
        std::abs(expected_mapping_issue_busy_ns) * 1e-12);
    if (!valid_mapping_wait(
            stats_.mapping_dram_wait_ops,
            stats_.mapping_dram_wait_ns,
            stats_.mapping_dram_wait_max_ns) ||
        !std::isfinite(stats_.mapping_dram_issue_busy_ns) ||
        std::abs(
            stats_.mapping_dram_issue_busy_ns -
            expected_mapping_issue_busy_ns) > mapping_busy_tolerance) {
        throw std::runtime_error(
            "HBF mapping DRAM telemetry diverged");
    }
    const auto write_buffer_dram_ops = checked_add(
        stats_.write_buffer_dram_read_ops,
        stats_.write_buffer_dram_write_ops,
        "HBF write-buffer DRAM access accounting");
    // A device-DRAM write buffer uses the device DRAM port, never the HBM
    // controller issue slot.
    const double expected_write_buffer_issue_busy_ns = write_buffer_on_device() ? 0.0 :
        static_cast<double>(write_buffer_dram_ops) *
        config_.host.ctrl_dram_issue_ns;
    const double write_buffer_busy_tolerance = std::max(
        1e-9,
        std::abs(expected_write_buffer_issue_busy_ns) * 1e-12);
    if (!valid_mapping_wait(
            stats_.write_buffer_dram_wait_ops,
            stats_.write_buffer_dram_wait_ns,
            stats_.write_buffer_dram_wait_max_ns) ||
        !std::isfinite(stats_.write_buffer_dram_issue_busy_ns) ||
        std::abs(
            stats_.write_buffer_dram_issue_busy_ns -
            expected_write_buffer_issue_busy_ns) >
            write_buffer_busy_tolerance ||
        (!config_.host.write_coalescing_enabled &&
         (write_buffer_dram_ops != 0 ||
          stats_.write_buffer_dram_read_bytes != 0 ||
          stats_.write_buffer_dram_write_bytes != 0))) {
        throw std::runtime_error(
            "HBF write-buffer DRAM telemetry diverged");
    }

    for (std::size_t plane_index = 0; plane_index < planes_.size(); ++plane_index) {
        const auto& plane = planes_.at(plane_index);
        for (const auto block_index : plane.free_blocks) {
            if (block_index >= blocks_.size() ||
                block_plane_index(block_index) != plane_index) {
                throw std::runtime_error(
                    "HBF free-block pool contains an out-of-plane block");
            }
            if (free_pool_membership.at(block_index) != 0) {
                throw std::runtime_error(
                    "HBF free-block pool contains a duplicate block");
            }
            const auto& block = blocks_.at(block_index);
            if (block.role != BlockRole::Free || block.erase_pending) {
                throw std::runtime_error(
                    "HBF free-block pool contains an owned or pending-erase block");
            }
            free_pool_membership[block_index] = 1;
        }
    }

    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        const auto& block = blocks_.at(block_index);
        if (block.role != BlockRole::StaticReadOnly) {
            // block_erases and block.erase_count both count an erase when it
            // is scheduled, so the two views agree in every observation
            // window without patching for in-flight erases.
            const auto effective_erase_count =
                static_cast<std::uint64_t>(block.erase_count);
            writable_blocks = checked_add(
                writable_blocks, 1, "HBF writable-block accounting");
            if (effective_erase_count != 0) {
                worn_blocks = checked_add(
                    worn_blocks, 1, "HBF worn-block accounting");
            }
            block_erase_count_sum = checked_add(
                block_erase_count_sum,
                effective_erase_count,
                "HBF block erase-count accounting");
            const auto erase_count =
                static_cast<long double>(effective_erase_count);
            block_erase_count_sum_squares += erase_count * erase_count;
            min_block_erase_count = std::min(
                min_block_erase_count,
                static_cast<std::uint32_t>(effective_erase_count));
            max_block_erase_count = std::max(
                max_block_erase_count,
                static_cast<std::uint32_t>(effective_erase_count));
            block_erase_count_histogram[
                static_cast<std::uint32_t>(effective_erase_count)]++;
        }
        const auto block_valid = static_cast<std::uint64_t>(block.valid_pages);
        const auto block_invalid = static_cast<std::uint64_t>(block.invalid_pages);
        const auto block_pending =
            static_cast<std::uint64_t>(block.pending_program_pages);
        const auto block_free = static_cast<std::uint64_t>(block.free_pages);
        if (block_valid > config_.device.pages_per_block ||
            block_invalid > config_.device.pages_per_block ||
            block_pending > config_.device.pages_per_block ||
            block_free > config_.device.pages_per_block) {
            throw std::runtime_error(
                "HBF block page-state counter exceeds block geometry");
        }

        std::uint64_t bitmap_valid = 0;
        if (block.valid_bitmap) {
            for (std::size_t index = 0; index < (config_.device.pages_per_block + 63) / 64; ++index) {
                const auto word = block.valid_bitmap[index];
                const auto first = index * 64;
                const auto width = block.next_page > first ?
                    std::min<std::uint64_t>(64, block.next_page - first) : 0;
                const auto allocated_mask = width == 64 ?
                    std::numeric_limits<std::uint64_t>::max() :
                    (std::uint64_t{1} << width) - 1;
                if ((word & ~allocated_mask) != 0) {
                    throw std::runtime_error(
                        "HBF valid-page bitmap extends beyond the allocated prefix");
                }
                bitmap_valid = checked_add(
                    bitmap_valid,
                    static_cast<std::uint64_t>(std::popcount(word)),
                    "HBF valid-page bitmap accounting");
            }
        }
        if (bitmap_valid != block_valid) {
            throw std::runtime_error(
                "HBF valid-page bitmap diverged from the block counter");
        }

        const bool allocatable_free =
            block.role == BlockRole::Free && !block.erase_pending;
        if (static_cast<bool>(free_pool_membership.at(block_index)) !=
            allocatable_free) {
            throw std::runtime_error(
                "HBF block role/pending-erase state diverged from its free pool");
        }

        if (block.role == BlockRole::StaticReadOnly) {
            if (block_invalid != 0 || block_pending != 0 || block_free != 0 ||
                block.next_page != config_.device.pages_per_block || block.erase_pending) {
                throw std::runtime_error(
                    "HBF static block has mutable or free page state");
            }
            static_unmaterialized_pages = checked_add(
                static_unmaterialized_pages,
                config_.device.pages_per_block - block_valid,
                "HBF static unmaterialized-page accounting");
        } else {
            if (block.role == BlockRole::RawPhysical) {
                raw_reserved_pages = checked_add(
                    raw_reserved_pages,
                    config_.device.pages_per_block,
                    "HBF raw reserved-page accounting audit");
            }
            const auto used = checked_add(
                checked_add(
                    block_valid,
                    block_invalid,
                    "HBF block valid/invalid accounting"),
                block_pending,
                "HBF block programmed/pending accounting");
            if (checked_add(
                    used, block_free, "HBF block total page accounting") !=
                    config_.device.pages_per_block ||
                block.next_page != used) {
                throw std::runtime_error(
                    "HBF block valid/invalid/pending/free pages do not conserve capacity");
            }
        }

        free_pages = checked_add(
            free_pages, block_free, "HBF global free-page accounting");
        valid_pages = checked_add(
            valid_pages, block_valid, "HBF global valid-page accounting");
        invalid_pages = checked_add(
            invalid_pages, block_invalid, "HBF global invalid-page accounting");
        pending_program_pages = checked_add(
            pending_program_pages,
            block_pending,
            "HBF global pending-program accounting");
        pending_mapping_publications = checked_add(
            pending_mapping_publications,
            block.pending_mapping_publications,
            "HBF global pending-mapping-publication accounting");
        auto& stack_free = free_pages_per_stack.at(stack_of_block(block_index));
        stack_free = checked_add(
            stack_free, block_free, "HBF per-stack free-page accounting");
    }

    std::uint64_t initial_raw_valid = 0;
    for (const auto& block : blocks_) {
        if (block.initial_raw_pages && (block.role != BlockRole::RawPhysical ||
            block.initial_raw_pages != config_.device.pages_per_block ||
            block.valid_pages != block.initial_raw_pages))
            throw std::runtime_error("compact raw initial block lost its full-zone validity");
        initial_raw_valid += block.initial_raw_pages;
    }
    std::uint64_t page_map_valid = 0;
    std::uint64_t page_map_pending = 0;
    for (const auto& [ppn, page] : programmed_pages_) {
        if (ppn >= total_pages_) {
            throw std::runtime_error("HBF page-state table contains an out-of-range PPN");
        }
        const auto block_index = static_cast<std::size_t>(
            ppn / config_.device.pages_per_block);
        const auto page_index = static_cast<std::uint32_t>(
            ppn % config_.device.pages_per_block);
        const auto& block = blocks_.at(block_index);
        if (block.initial_raw_pages)
            throw std::runtime_error("materialized page overlaps compact raw initial block");
        if (page_index >= block.next_page) {
            throw std::runtime_error(
                "HBF page-state entry is outside the allocated block prefix");
        }
        switch (page.status) {
        case PageStatus::Erased:
            if (block.is_valid(page_index)) {
                throw std::runtime_error(
                    "HBF pending page-state entry is already valid in its block bitmap");
            }
            page_map_pending++;
            break;
        case PageStatus::StaticReadOnly:
        case PageStatus::Valid:
            if (!block.is_valid(page_index)) {
                throw std::runtime_error(
                    "HBF valid page-state entry is absent from its block bitmap");
            }
            page_map_valid++;
            break;
        }
    }
    std::uint64_t compact_valid_pages = 0;
    if (compact_logical_image_) {
        const auto& image = *compact_logical_image_;
        if (image.data_blocks_by_plane.size() != planes_.size() ||
            image.mapping_ppns.size() != image.vpn_slot_count ||
            image.vpn_ranges.size() != image.vpn_slot_count ||
            image.vpn_offsets_by_stack.size() != config_.device.stacks) {
            throw std::runtime_error(
                "HBF compact image directory dimensions are inconsistent");
        }
        std::vector<std::uint8_t> compact_data_block_seen(blocks_.size(), 0);
        std::uint64_t compact_live_data_pages = 0;
        for (std::size_t plane = 0;
             plane < image.data_blocks_by_plane.size();
             ++plane) {
            const auto& assigned = image.data_blocks_by_plane[plane];
            for (std::size_t ordinal = 0;
                 ordinal < assigned.size();
                 ++ordinal) {
                const auto block_number = assigned[ordinal];
                if (block_number >= blocks_.size() ||
                    block_plane_index(static_cast<std::size_t>(block_number)) != plane ||
                    compact_data_block_seen.at(
                        static_cast<std::size_t>(block_number)) != 0) {
                    throw std::runtime_error(
                        "HBF compact data-block directory is invalid or duplicated");
                }
                compact_data_block_seen[static_cast<std::size_t>(block_number)] = 1;
                const auto location =
                    image.data_block_locations.find(block_number);
                if (location == image.data_block_locations.end() ||
                    location->second.plane != plane ||
                    location->second.block_ordinal != ordinal) {
                    throw std::runtime_error(
                        "HBF compact data-block reverse index diverged");
                }
                const auto& block = blocks_.at(
                    static_cast<std::size_t>(block_number));
                const auto live = image.live_data_pages_by_block.find(
                    block_number);
                const auto live_pages =
                    live == image.live_data_pages_by_block.end() ?
                    0 : static_cast<std::uint64_t>(live->second);
                if (live_pages > block.valid_pages ||
                    (live_pages != 0 &&
                     (block.role != BlockRole::Data ||
                      block.erase_pending))) {
                    throw std::runtime_error(
                        "HBF compact data-block live count references "
                        "non-live physical state");
                }
                compact_live_data_pages = checked_add(
                    compact_live_data_pages,
                    live_pages,
                    "HBF compact live data-page accounting");
            }
        }
        for (const auto& [block_number, live_pages] :
             image.live_data_pages_by_block) {
            if (live_pages == 0 ||
                block_number >= compact_data_block_seen.size() ||
                compact_data_block_seen[
                    static_cast<std::size_t>(block_number)] == 0) {
                throw std::runtime_error(
                    "HBF compact live data-block index is not canonical");
            }
        }
        if (image.retired_lpns.size() > image.page_count ||
            compact_live_data_pages !=
                image.page_count - image.retired_lpns.size()) {
            throw std::runtime_error(
                "HBF compact live/retired data pages do not conserve the "
                "logical image");
        }
        std::unordered_set<std::uint64_t> compact_mapping_seen;
        compact_mapping_seen.reserve(
            static_cast<std::size_t>(image.mapping_page_count));
        std::uint64_t compact_range_pages = 0;
        std::uint64_t compact_live_mapping_pages = 0;
        for (std::size_t index = 0;
             index < image.mapping_ppns.size();
             ++index) {
            const auto& range = image.vpn_ranges[index];
            const auto& compact_ppn = image.mapping_ppns[index];
            const bool needs_mapping = range.page_count != 0;
            if (needs_mapping != compact_ppn.has_value()) {
                throw std::runtime_error(
                    "HBF compact mapping slot activity is inconsistent");
            }
            compact_range_pages = checked_add(
                compact_range_pages,
                range.page_count,
                "HBF compact mapping-range page accounting");
            if (!compact_ppn) {
                continue;
            }
            const auto ppn = *compact_ppn;
            if (ppn >= total_pages_) {
                throw std::runtime_error(
                    "HBF compact mapping-page directory is out of range");
            }
            const auto block_index = static_cast<std::size_t>(
                ppn / config_.device.pages_per_block);
            const auto page_index = static_cast<std::uint32_t>(
                ppn % config_.device.pages_per_block);
            const auto mapping_vpn = checked_add(
                image.first_vpn,
                index,
                "HBF compact accounting mapping VPN");
            const auto inverse = image.mapping_vpn_by_ppn.find(ppn);
            if (!compact_mapping_seen.insert(ppn).second ||
                inverse == image.mapping_vpn_by_ppn.end() ||
                inverse->second != mapping_vpn) {
                throw std::runtime_error(
                    "HBF compact mapping-page reverse index diverged");
            }
            if (!image.retired_mapping_vpns.contains(mapping_vpn)) {
                const auto& block = blocks_.at(block_index);
                if (block.role != BlockRole::Mapping ||
                    block.erase_pending ||
                    !block.is_valid(page_index)) {
                    throw std::runtime_error(
                        "HBF compact mapping-page directory references a "
                        "non-live page");
                }
                ++compact_live_mapping_pages;
            }
        }
        if (compact_range_pages != image.page_count ||
            compact_mapping_seen.size() != image.mapping_page_count ||
            image.mapping_vpn_by_ppn.size() != image.mapping_page_count ||
            image.retired_mapping_vpns.size() >
                image.mapping_page_count ||
            compact_live_mapping_pages !=
                image.mapping_page_count -
                    image.retired_mapping_vpns.size()) {
            throw std::runtime_error(
                "HBF compact mapping slots do not conserve the logical image");
        }
        std::uint64_t indexed_live_mapping_pages = 0;
        for (const auto& [block_number, live_pages] :
             image.live_mapping_pages_by_block) {
            if (live_pages == 0 || block_number >= blocks_.size()) {
                throw std::runtime_error(
                    "HBF compact live mapping-block index is invalid");
            }
            indexed_live_mapping_pages = checked_add(
                indexed_live_mapping_pages,
                live_pages,
                "HBF compact indexed live mapping pages");
        }
        if (indexed_live_mapping_pages != compact_live_mapping_pages) {
            throw std::runtime_error(
                "HBF compact mapping live-page block index diverged");
        }
        compact_valid_pages = checked_add(
            compact_live_data_pages,
            compact_live_mapping_pages,
            "HBF compact valid-page accounting");
        stats_.compact_live_logical_data_pages =
            compact_live_data_pages;
        stats_.compact_live_mapping_pages =
            compact_live_mapping_pages;
        stats_.compact_retired_logical_data_pages =
            image.retired_lpns.size();
        stats_.compact_retired_mapping_pages =
            image.retired_mapping_vpns.size();
    } else {
        stats_.compact_live_logical_data_pages = 0;
        stats_.compact_live_mapping_pages = 0;
        stats_.compact_retired_logical_data_pages = 0;
        stats_.compact_retired_mapping_pages = 0;
    }
    // Compact initial counts describe the original dense address image,
    // including retired slots carried through a persistent-image restart.
    // initial_* counts describe live data at this run's setup. Deallocation
    // can make the latter smaller; only live + retired must conserve the
    // historical compact image.
    if (checked_add(
            stats_.compact_live_logical_data_pages,
            stats_.compact_retired_logical_data_pages,
            "HBF compact initial data-page lifecycle") !=
            stats_.compact_initial_logical_data_pages ||
        checked_add(
            stats_.compact_live_mapping_pages,
            stats_.compact_retired_mapping_pages,
            "HBF compact initial mapping-page lifecycle") !=
            stats_.compact_initial_mapping_pages) {
        throw std::runtime_error(
            "HBF compact initial-image lifecycle accounting diverged");
    }
    // Every allocated prefix was checked against its bitmap and counters
    // above. Live/compact entries and pending reservations account for all
    // non-invalid positions; the remaining prefix is invalid, never free.
    // No per-invalid-page identity or extra bitmap is needed for this audit.
    if (checked_add(
            page_map_valid + initial_raw_valid,
            compact_valid_pages,
            "HBF materialized/compact valid-page accounting") != valid_pages ||
        page_map_pending != pending_program_pages) {
        throw std::runtime_error(
            "HBF page-state table diverged from block page counters");
    }
    const auto validate_mapping_target = [this](
                                             std::uint64_t ppn,
                                             std::uint64_t expected_lpn,
                                             PageOwner expected_owner,
                                             const char* name) {
        const auto page = programmed_pages_.find(ppn);
        if (ppn >= total_pages_ || page == programmed_pages_.end() ||
            page->second.status != PageStatus::Valid ||
            page->second.owner != expected_owner ||
            page->second.lpn != expected_lpn) {
            throw std::runtime_error(
                std::string("HBF ") + name +
                " points to a non-live or wrongly owned physical page");
        }
    };
    for (const auto& [lpn, ppn] : lpn_to_ppn_) {
        validate_mapping_target(ppn, lpn, PageOwner::Logical, "L2P mapping");
    }
    for (const auto& [mapping_vpn, ppn] : mapping_vpn_to_ppn_) {
        validate_mapping_target(
            ppn,
            metadata_lpn(mapping_vpn),
            PageOwner::Mapping,
            "mapping-page directory");
    }
    // Between media-program and mapping-publication callbacks, both old and
    // new versions may legitimately be valid. Once all program/publication
    // pins are gone, every live logical or mapping page must be reachable in
    // the corresponding directory; this catches orphaned GC relocations.
    if (pending_program_pages == 0 && pending_mapping_publications == 0) {
        for (const auto& [ppn, page] : programmed_pages_) {
            if (page.status != PageStatus::Valid) {
                continue;
            }
            if (page.owner == PageOwner::Logical) {
                const auto mapping = lpn_to_ppn_.find(page.lpn);
                if (mapping == lpn_to_ppn_.end() || mapping->second != ppn) {
                    throw std::runtime_error(
                        "HBF quiescent logical page is unreachable from L2P");
                }
            } else if (page.owner == PageOwner::Mapping) {
                const auto mapping_vpn = metadata_vpn(page.lpn);
                const auto mapping = mapping_vpn_to_ppn_.find(mapping_vpn);
                if (mapping == mapping_vpn_to_ppn_.end() || mapping->second != ppn) {
                    throw std::runtime_error(
                        "HBF quiescent mapping page is unreachable from its directory");
                }
            } else if (page.owner != PageOwner::RawPhysical) {
                throw std::runtime_error(
                    "HBF valid page has an invalid owner at quiescence");
            }
        }
    }
    if (free_pages != free_pages_ || free_pages_per_stack != free_pages_per_stack_) {
        throw std::runtime_error(
            "HBF global/per-stack free-page counters do not match block state");
    }
    const auto accounted_pages = checked_add(
        checked_add(
            checked_add(
                free_pages, valid_pages, "HBF free/valid capacity accounting"),
            invalid_pages,
            "HBF free/valid/invalid capacity accounting"),
        checked_add(
            pending_program_pages,
            static_unmaterialized_pages,
            "HBF pending/static capacity accounting"),
        "HBF total physical-page accounting");
    if (accounted_pages != total_pages_) {
        throw std::runtime_error(
            "HBF physical page states do not conserve total capacity");
    }
    if (block_erase_count_sum != checked_add(
            restored_block_erases_,
            stats_.block_erases,
            "HBF restored/current block erase accounting")) {
        throw std::runtime_error(
            "HBF per-block erase counts diverged from block_erases");
    }

    const auto classified_programs = checked_add(
        checked_add(
            checked_add(
                stats_.data_programs,
                stats_.mapping_page_programs,
                "HBF data/mapping program accounting"),
            stats_.gc_relocations,
            "HBF classified program accounting"),
        stats_.static_wear_leveling_relocations,
        "HBF wear-leveling program accounting");
    const auto classified_payload_bytes = checked_add(
        checked_add(
            checked_add(
                stats_.data_program_payload_bytes,
                stats_.mapping_program_payload_bytes,
                "HBF data/mapping payload-byte accounting"),
            stats_.gc_relocation_payload_bytes,
            "HBF classified payload-byte accounting"),
        stats_.static_wear_leveling_relocation_payload_bytes,
        "HBF wear-leveling payload-byte accounting");
    std::uint64_t active_gc_relocations = 0;
    std::uint64_t active_wear_relocations = 0;
    for (std::size_t stack = 0; stack < config_.device.stacks; ++stack) {
        if (const auto& victim = gc_victim_by_stack_.at(stack)) {
            active_gc_relocations = checked_add(
                active_gc_relocations, victim->relocated_pages,
                "HBF active GC relocation accounting");
        }
        if (const auto& migration = wear_leveling_by_stack_.at(stack)) {
            active_wear_relocations = checked_add(
                active_wear_relocations, migration->relocated_pages,
                "HBF active wear-leveling relocation accounting");
        }
    }
    std::uint64_t pending_relocation_publications = 0;
    for (const auto* pending_updates : {&pending_lpn_updates_, &pending_vpn_updates_}) {
        for (const auto& entry : *pending_updates) {
            for (const auto& update : entry.second) {
                if (update.relocation &&
                    pending_commit_time_by_sequence_.contains(update.sequence)) {
                    pending_relocation_publications = checked_add(
                        pending_relocation_publications, 1,
                        "HBF pending relocation publication accounting");
                }
            }
        }
    }
    if (stats_.page_programs != classified_programs ||
        stats_.raw_physical_programs > stats_.data_programs ||
        stats_.raw_physical_program_payload_bytes != checked_mul(
            stats_.raw_physical_programs,
            config_.device.page_size_bytes,
            "HBF raw physical program payload-byte accounting") ||
        stats_.raw_physical_program_payload_bytes >
            stats_.data_program_payload_bytes ||
        stats_.data_program_payload_bytes != checked_mul(
            stats_.data_programs,
            config_.device.page_size_bytes,
            "HBF data-program payload-byte accounting") ||
        stats_.mapping_program_payload_bytes != checked_mul(
            stats_.mapping_page_programs,
            config_.device.page_size_bytes,
            "HBF mapping-program payload-byte accounting") ||
        stats_.gc_relocation_payload_bytes != checked_mul(
            stats_.gc_relocations,
            config_.device.page_size_bytes,
            "HBF GC-relocation payload-byte accounting") ||
        stats_.static_wear_leveling_relocation_payload_bytes != checked_mul(
            stats_.static_wear_leveling_relocations,
            config_.device.page_size_bytes,
            "HBF wear-leveling relocation payload-byte accounting") ||
        checked_add(
            stats_.static_wear_leveling_relocations,
            stats_.static_wear_leveling_reclaimed_invalid_pages,
            "HBF wear-leveling relocated/reclaimed page accounting") !=
            checked_add(
                checked_mul(
                    stats_.static_wear_leveling_runs,
                    config_.device.pages_per_block,
                    "HBF wear-leveling victim-page accounting"),
                active_wear_relocations,
                "HBF completed/active wear-leveling accounting") ||
        stats_.physical_write_bytes != classified_payload_bytes ||
        stats_.physical_write_bytes != checked_mul(
            stats_.page_programs,
            config_.device.page_size_bytes,
            "HBF physical-write byte accounting") ||
        stats_.physical_read_bytes != checked_mul(
            stats_.page_reads,
            config_.device.page_size_bytes,
            "HBF physical-read byte accounting")) {
        throw std::runtime_error(
            "HBF program/read byte accounting identities diverged");
    }
    if (stats_.gc_relocations != checked_add(
            stats_.gc_data_relocations,
            stats_.gc_mapping_relocations,
            "HBF GC relocation owner accounting")) {
        throw std::runtime_error(
            "HBF GC data/mapping relocation counts do not conserve relocations");
    }
    const auto gc_victim_pages = checked_add(
        checked_mul(stats_.gc_runs, config_.device.pages_per_block,
            "HBF GC victim-page accounting"),
        active_gc_relocations,
        "HBF completed/active GC accounting");
    if (checked_add(
            stats_.gc_relocations,
            stats_.gc_reclaimed_invalid_pages,
            "HBF GC relocated/reclaimed page accounting") != gc_victim_pages ||
        stats_.block_erases != checked_add(stats_.erase_requests, stats_.auto_erase_requests,
            "HBF explicit/automatic erase accounting") ||
        checked_add(
            stats_.invalidations, pending_relocation_publications,
            "HBF committed/pending relocation invalidation accounting") < checked_add(
            stats_.gc_relocations,
            stats_.static_wear_leveling_relocations,
            "HBF relocation invalidation accounting")) {
        throw std::runtime_error(
            "HBF GC victim, erase, or invalidation accounting diverged");
    }
    const auto flash_transactions = checked_add(
        checked_add(
            stats_.page_reads,
            stats_.page_programs,
            "HBF flash read/program accounting"),
        stats_.block_erases,
        "HBF flash transaction accounting");
    if (stats_.flash_scheduler_enqueues != flash_transactions ||
        stats_.flash_scheduler_issues != flash_transactions) {
        throw std::runtime_error(
            "HBF flash scheduler counts do not conserve media transactions");
    }

    stats_.total_pages = total_pages_;
    stats_.free_pages = free_pages;
    stats_.valid_pages = valid_pages;
    stats_.invalid_pages = invalid_pages;
    stats_.pending_program_pages = pending_program_pages;
    stats_.pending_mapping_publications = pending_mapping_publications;
    stats_.static_unmaterialized_pages = static_unmaterialized_pages;
    stats_.raw_reserved_pages = raw_reserved_pages;
    for (std::size_t stack = 0; stack < config_.device.stacks; ++stack) {
        if (active_planes_in_stack(stack) != scan_active_planes_in_stack(stack)) {
            throw std::runtime_error(
                "HBF active-plane counters diverged from block roles");
        }
    }
    stats_.accounting_verified = true;
    stats_.writable_blocks = writable_blocks;
    stats_.writable_pages = checked_mul(
        writable_blocks,
        config_.device.pages_per_block,
        "HBF writable-page accounting");
    stats_.writable_payload_bytes = checked_mul(
        stats_.writable_pages,
        config_.device.page_size_bytes,
        "HBF writable-payload accounting");
    stats_.worn_blocks = worn_blocks;
    stats_.block_erase_count_sum = block_erase_count_sum;
    stats_.block_erase_count_sum_squares = block_erase_count_sum_squares;
    stats_.min_block_erase_count = writable_blocks == 0 ?
        0 : min_block_erase_count;
    stats_.max_block_erase_count = max_block_erase_count;
    stats_.block_erase_count_histogram = std::move(block_erase_count_histogram);
}

void HbfController::refresh_parallel_stats() const {
    refresh_accounting_stats();
    stats_.mapping_entries = logical_mapping_entry_count();
    stats_.thermal_peak_temperature_c = 0.0;
    stats_.thermal_final_temperature_c = 0.0;
    stats_.thermal_throttled_stacks = 0;
    if (config_.device.thermal_enabled) {
        // Project every stack to the finish frontier on a copy: reading
        // stats must never advance governor state.
        const double horizon_ns = std::max(stats_.finish_ns, 0.0);
        double peak_c = thermal_boot_temperature_c_;
        double final_c = thermal_boot_temperature_c_;
        std::uint64_t throttled_stacks = 0;
        for (const auto& stack : thermal_stacks_) {
            auto node = stack.node;
            thermal_advance(node, horizon_ns, false);
            peak_c = std::max(peak_c, node.peak_c);
            final_c = std::max(final_c, node.temperature_c);
            throttled_stacks += node.throttled ? 1 : 0;
        }
        stats_.thermal_peak_temperature_c = peak_c;
        stats_.thermal_final_temperature_c = final_c;
        stats_.thermal_throttled_stacks = throttled_stacks;
    }
    // The same physical work is accumulated in request order for the public
    // total and in resource order for directional/per-die totals. Their
    // round-off bound grows with the number of additions; a fixed 32-epsilon
    // tolerance falsely rejected long, otherwise exact replays.
    const auto accumulation_terms = std::max<std::uint64_t>(
        32,
        checked_add(
            checked_add(
                checked_add(
                    stats_.read_requests,
                    stats_.program_requests,
                    "HBF conservation request terms"),
                stats_.erase_requests,
                "HBF conservation request/erase terms"),
            checked_add(
                checked_add(
                    stats_.page_reads,
                    stats_.page_programs,
                    "HBF conservation media terms"),
                checked_add(
                    stats_.read_buffer_hits,
                    stats_.write_buffer_read_hits,
                    "HBF conservation buffer terms"),
                "HBF conservation media/buffer terms"),
            "HBF conservation total terms"));
    const auto work_conserved = [accumulation_terms](
                                    double total, double decode, double encode) {
        const double expected = decode + encode;
        const double scale = std::max({1.0, std::abs(total), std::abs(expected)});
        return std::abs(total - expected) <=
            8.0 * static_cast<double>(accumulation_terms) *
                std::numeric_limits<double>::epsilon() * scale;
    };
    if (stats_.ecc_decode_ops != stats_.page_reads) {
        throw std::runtime_error(
            "HBF ECC decode operation count diverged from physical page reads");
    }
    if (stats_.ecc_encode_ops != stats_.page_programs) {
        throw std::runtime_error(
            "HBF ECC encode operation count diverged from physical page programs");
    }
    const auto expected_decode_bytes = checked_mul(
        stats_.ecc_decode_ops, page_wire_bytes(), "HBF ECC decode conservation");
    const auto expected_encode_bytes = checked_mul(
        stats_.ecc_encode_ops, page_wire_bytes(), "HBF ECC encode conservation");
    if (stats_.ecc_decode_codeword_bytes != expected_decode_bytes ||
        stats_.ecc_encode_codeword_bytes != expected_encode_bytes ||
        stats_.ecc_codeword_bytes != checked_add(
            expected_decode_bytes, expected_encode_bytes,
            "HBF ECC total conservation")) {
        throw std::runtime_error(
            "HBF ECC codeword-byte accounting did not conserve page operations");
    }
    if (!work_conserved(
            stats_.stage_work.ecc_queue_wait_ns,
            stats_.ecc_decode_queue_wait_ns,
            stats_.ecc_encode_queue_wait_ns) ||
        !work_conserved(
            stats_.stage_work.ecc_latency_ns,
            stats_.ecc_decode_latency_work_ns,
            stats_.ecc_encode_latency_work_ns) ||
        !work_conserved(
            stats_.ecc_issue_busy_ns,
            stats_.ecc_decode_issue_busy_ns,
            stats_.ecc_encode_issue_busy_ns)) {
        throw std::runtime_error(
            "HBF ECC directional work accounting did not conserve totals");
    }
    stats_.stacks = config_.device.stacks;
    stats_.hbio_data_resources = 2ULL * config_.device.stacks * config_.device.channels_per_stack;
    stats_.hbio_command_resources = config_.device.stacks * config_.device.channels_per_stack;
    stats_.channels = channels_.size();
    stats_.dies = dies_.size();
    stats_.planes = planes_.size();
    stats_.media_lanes = checked_mul(
        static_cast<std::uint64_t>(planes_.size()),
        config_.device.media_lanes_per_plane,
        "HBF stats media_lanes");
    stats_.subarrays = checked_mul(
        static_cast<std::uint64_t>(planes_.size()),
        subarrays_per_plane_,
        "HBF stats subarrays");
    stats_.page_buffer_banks = checked_mul(
        static_cast<std::uint64_t>(planes_.size()),
        config_.device.page_buffer_banks_per_plane,
        "HBF stats page_buffer_banks");
    stats_.active_channels = 0;
    stats_.active_dies = 0;
    stats_.active_planes = 0;
    stats_.active_media_lanes = 0;
    stats_.active_subarrays = 0;
    stats_.active_page_buffer_banks = 0;
    stats_.max_plane_ops = 0;
    stats_.max_media_lane_reads = 0;
    stats_.max_subarray_reads = 0;
    stats_.max_page_buffer_bank_reads = 0;
    stats_.max_die_transactions = 0;
    stats_.max_plane_media_busy_ns = 0.0;
    stats_.avg_active_plane_media_busy_ns = 0.0;
    stats_.avg_active_plane_ops = 0.0;
    stats_.read_lane_busy_ns = 0.0;
    stats_.subarray_read_busy_ns = 0.0;
    stats_.page_buffer_bank_busy_ns = 0.0;
    stats_.max_media_lane_busy_ns = 0.0;
    stats_.avg_active_media_lane_busy_ns = 0.0;
    stats_.avg_active_media_lane_reads = 0.0;
    stats_.max_subarray_busy_ns = 0.0;
    stats_.avg_active_subarray_busy_ns = 0.0;
    stats_.avg_active_subarray_reads = 0.0;
    stats_.max_page_buffer_bank_busy_ns = 0.0;
    stats_.avg_active_page_buffer_bank_busy_ns = 0.0;
    stats_.avg_active_page_buffer_bank_reads = 0.0;
    stats_.max_channel_busy_ns = 0.0;
    stats_.avg_active_channel_busy_ns = 0.0;
    stats_.avg_active_die_transactions = 0.0;
    stats_.sequencer_busy_ns = 0.0;
    stats_.hb_io_command_busy_ns = 0.0;
    stats_.hb_io_data_busy_ns = 0.0;
    stats_.logic_ingress_busy_ns = 0.0;
    stats_.tsv_busy_ns = 0.0;
    stats_.sram_busy_ns = 0.0;
    stats_.flash_source_queue_busy_ns = 0.0;
    stats_.channel_command_busy_ns = 0.0;
    stats_.channel_data_busy_ns = 0.0;
    stats_.logic_ingress_resources = logic_dies_.size();
    stats_.tsv_resources = logic_dies_.size();
    stats_.sram_resources = logic_dies_.size();
    stats_.flash_source_queue_resources = checked_mul(
        static_cast<std::uint64_t>(dies_.size()),
        static_cast<std::uint64_t>(std::tuple_size_v<
            decltype(DieState::source_queues)>),
        "HBF stats flash source queue resources");
    stats_.channel_command_resources = channels_.size();
    stats_.channel_data_resources = channels_.size();
    stats_.active_ecc_dies = 0;
    stats_.max_ecc_inflight_per_die = 0;
    stats_.max_ecc_issue_busy_ns = 0.0;
    stats_.avg_active_ecc_issue_busy_ns = 0.0;

    double active_plane_busy_ns = 0.0;
    std::uint64_t active_plane_ops = 0;
    for (const auto& plane : planes_) {
        const auto ops = plane.read_count + plane.program_count + plane.erase_count;
        if (ops == 0 && plane.media_busy_ns <= 0.0) {
            continue;
        }
        stats_.active_planes++;
        active_plane_busy_ns += plane.media_busy_ns;
        active_plane_ops += ops;
        stats_.max_plane_media_busy_ns =
            std::max(stats_.max_plane_media_busy_ns, plane.media_busy_ns);
        stats_.max_plane_ops = std::max(stats_.max_plane_ops, ops);

        for (const auto& lane : plane.media_lanes) {
            if (lane.read_count == 0 && lane.busy_ns <= 0.0) {
                continue;
            }
            stats_.active_media_lanes++;
            stats_.read_lane_busy_ns += lane.busy_ns;
            stats_.max_media_lane_busy_ns =
                std::max(stats_.max_media_lane_busy_ns, lane.busy_ns);
            stats_.max_media_lane_reads =
                std::max(stats_.max_media_lane_reads, lane.read_count);
        }

        for (const auto& subarray : plane.subarrays) {
            if (subarray.read_count == 0 && subarray.busy_ns <= 0.0) {
                continue;
            }
            stats_.active_subarrays++;
            stats_.subarray_read_busy_ns += subarray.busy_ns;
            stats_.max_subarray_busy_ns =
                std::max(stats_.max_subarray_busy_ns, subarray.busy_ns);
            stats_.max_subarray_reads =
                std::max(stats_.max_subarray_reads, subarray.read_count);
        }

        for (const auto& bank : plane.page_buffer_banks) {
            if (bank.read_count == 0 && bank.busy_ns <= 0.0) {
                continue;
            }
            stats_.active_page_buffer_banks++;
            stats_.page_buffer_bank_busy_ns += bank.busy_ns;
            stats_.max_page_buffer_bank_busy_ns =
                std::max(stats_.max_page_buffer_bank_busy_ns, bank.busy_ns);
            stats_.max_page_buffer_bank_reads =
                std::max(stats_.max_page_buffer_bank_reads, bank.read_count);
        }
    }
    stats_.media_busy_ns = active_plane_busy_ns;
    if (stats_.active_planes != 0) {
        stats_.avg_active_plane_media_busy_ns =
            active_plane_busy_ns / static_cast<double>(stats_.active_planes);
        stats_.avg_active_plane_ops =
            static_cast<double>(active_plane_ops) / static_cast<double>(stats_.active_planes);
    }
    if (stats_.active_media_lanes != 0) {
        stats_.avg_active_media_lane_busy_ns =
            stats_.read_lane_busy_ns / static_cast<double>(stats_.active_media_lanes);
        std::uint64_t active_lane_reads = 0;
        for (const auto& plane : planes_) {
            for (const auto& lane : plane.media_lanes) {
                if (lane.read_count != 0 || lane.busy_ns > 0.0) {
                    active_lane_reads += lane.read_count;
                }
            }
        }
        stats_.avg_active_media_lane_reads =
            static_cast<double>(active_lane_reads) /
            static_cast<double>(stats_.active_media_lanes);
    }
    if (stats_.active_subarrays != 0) {
        stats_.avg_active_subarray_busy_ns =
            stats_.subarray_read_busy_ns / static_cast<double>(stats_.active_subarrays);
        std::uint64_t active_subarray_reads = 0;
        for (const auto& plane : planes_) {
            for (const auto& subarray : plane.subarrays) {
                if (subarray.read_count != 0 || subarray.busy_ns > 0.0) {
                    active_subarray_reads += subarray.read_count;
                }
            }
        }
        stats_.avg_active_subarray_reads =
            static_cast<double>(active_subarray_reads) /
            static_cast<double>(stats_.active_subarrays);
    }
    if (stats_.active_page_buffer_banks != 0) {
        double active_bank_busy_ns = 0.0;
        std::uint64_t active_bank_reads = 0;
        for (const auto& plane : planes_) {
            for (const auto& bank : plane.page_buffer_banks) {
                if (bank.read_count != 0 || bank.busy_ns > 0.0) {
                    active_bank_busy_ns += bank.busy_ns;
                    active_bank_reads += bank.read_count;
                }
            }
        }
        stats_.avg_active_page_buffer_bank_busy_ns =
            active_bank_busy_ns / static_cast<double>(stats_.active_page_buffer_banks);
        stats_.avg_active_page_buffer_bank_reads =
            static_cast<double>(active_bank_reads) /
            static_cast<double>(stats_.active_page_buffer_banks);
    }

    double active_channel_busy_ns = 0.0;
    for (const auto& channel : channels_) {
        stats_.channel_command_busy_ns += channel.command_busy_ns;
        stats_.channel_data_busy_ns += channel.data_busy_ns;
        const double busy_ns = channel.command_busy_ns + channel.data_busy_ns;
        if (channel.command_count == 0 && channel.data_count == 0 && busy_ns <= 0.0) {
            continue;
        }
        stats_.active_channels++;
        active_channel_busy_ns += busy_ns;
        stats_.max_channel_busy_ns = std::max(stats_.max_channel_busy_ns, busy_ns);
    }
    if (stats_.active_channels != 0) {
        stats_.avg_active_channel_busy_ns =
            active_channel_busy_ns / static_cast<double>(stats_.active_channels);
    }
    if (!work_conserved(
            stats_.stage_work.channel_transfer_ns,
            stats_.channel_command_busy_ns,
            stats_.channel_data_busy_ns)) {
        throw std::runtime_error(
            "HBF channel command/data resource work did not conserve stage work");
    }

    for (const auto& logic_die : logic_dies_) {
        stats_.logic_ingress_busy_ns += logic_die.ingress.reserved_work_ns;
        stats_.tsv_busy_ns += logic_die.tsv.reserved_work_ns;
        stats_.sram_busy_ns += logic_die.sram.reserved_work_ns;
    }
    for (const auto& channel : media_->channels()) {
        stats_.hb_io_command_busy_ns += channel.command_work_ns;
        stats_.hb_io_data_busy_ns += channel.rx_work_ns + channel.tx_work_ns;
    }
    if (!work_conserved(
            stats_.stage_work.hb_io_transfer_ns,
            stats_.hb_io_command_busy_ns,
            stats_.hb_io_data_busy_ns)) {
        throw std::runtime_error(
            "HBF HBIO command/data work accounting did not conserve totals");
    }
    if (!work_conserved(
            stats_.stage_work.tsv_transfer_ns,
            stats_.tsv_busy_ns,
            0.0)) {
        throw std::runtime_error(
            "HBF TSV resource work did not conserve stage work");
    }
    if (!work_conserved(
            stats_.stage_work.sram_staging_ns,
            stats_.sram_busy_ns,
            0.0)) {
        throw std::runtime_error(
            "HBF SRAM resource work did not conserve stage work");
    }

    std::uint64_t active_die_transactions = 0;
    double active_ecc_issue_busy_ns = 0.0;
    std::uint64_t die_ecc_decode_ops = 0;
    std::uint64_t die_ecc_encode_ops = 0;
    double die_ecc_issue_busy_ns = 0.0;
    for (const auto& die : dies_) {
        stats_.sequencer_busy_ns += die.sequencer_busy_ns;
        for (const auto& queue : die.source_queues) {
            stats_.flash_source_queue_busy_ns += queue.reserved_work_ns;
        }
        die_ecc_decode_ops = checked_add(
            die_ecc_decode_ops, die.ecc_decode_ops, "HBF die ECC decode ops");
        die_ecc_encode_ops = checked_add(
            die_ecc_encode_ops, die.ecc_encode_ops, "HBF die ECC encode ops");
        die_ecc_issue_busy_ns += die.ecc_issue_busy_ns;
        if (die.ecc_decode_ops != 0 || die.ecc_encode_ops != 0 ||
            die.ecc_issue_busy_ns > 0.0) {
            stats_.active_ecc_dies++;
            active_ecc_issue_busy_ns += die.ecc_issue_busy_ns;
            stats_.max_ecc_issue_busy_ns =
                std::max(stats_.max_ecc_issue_busy_ns, die.ecc_issue_busy_ns);

            // Exact reservation calendars may backfill issue slots, so call
            // order is not time order. Sweep the retained latency intervals
            // and combine them with the folded peak of retired ones.
            std::uint64_t max_inflight = ecc_max_inflight(die);
            stats_.max_ecc_inflight_per_die =
                std::max(stats_.max_ecc_inflight_per_die, max_inflight);
        }
        if (die.transaction_count == 0 && die.sequencer_busy_ns <= 0.0) {
            continue;
        }
        stats_.active_dies++;
        active_die_transactions += die.transaction_count;
        stats_.max_die_transactions =
            std::max(stats_.max_die_transactions, die.transaction_count);
    }
    if (stats_.active_dies != 0) {
        stats_.avg_active_die_transactions =
            static_cast<double>(active_die_transactions) / static_cast<double>(stats_.active_dies);
    }
    if (stats_.active_ecc_dies != 0) {
        stats_.avg_active_ecc_issue_busy_ns =
            active_ecc_issue_busy_ns / static_cast<double>(stats_.active_ecc_dies);
    }
    if (die_ecc_decode_ops != stats_.ecc_decode_ops ||
        die_ecc_encode_ops != stats_.ecc_encode_ops ||
        !work_conserved(stats_.ecc_issue_busy_ns, die_ecc_issue_busy_ns, 0.0)) {
        throw std::runtime_error(
            "HBF ECC per-die accounting did not conserve aggregate work");
    }
    if (stats_.read_splits > stats_.read_requests) {
        throw std::runtime_error(
            "HBF read-split count exceeds read-request count");
    }
    if (stats_.scalar_read_requests != stats_.read_requests) {
        throw std::runtime_error("HBF page-read accounting did not conserve requests");
    }
    const auto expected_page_admissions = checked_add(
        stats_.read_split_pages,
        stats_.read_requests - stats_.read_splits,
        "HBF expected page-read admissions");
    if (stats_.page_read_admission_events != expected_page_admissions ||
        stats_.page_read_admission_waited_pages >
            stats_.page_read_admission_events ||
        (stats_.page_read_admission_waited_pages == 0) !=
            (stats_.page_read_admission_wait_ns == 0.0) ||
        (stats_.page_read_admission_waited_pages == 0) !=
            (stats_.page_read_admission_max_wait_ns == 0.0) ||
        stats_.page_read_admission_max_wait_ns >
            stats_.page_read_admission_wait_ns) {
        throw std::runtime_error(
            "HBF page-read admission accounting did not conserve requests: events=" +
            std::to_string(stats_.page_read_admission_events) + " expected=" +
            std::to_string(expected_page_admissions) + " waited=" +
            std::to_string(stats_.page_read_admission_waited_pages) + " wait=" +
            std::to_string(stats_.page_read_admission_wait_ns) + " max=" +
            std::to_string(stats_.page_read_admission_max_wait_ns));
    }
}

} // namespace hbfsim::host
