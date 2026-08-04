#include "physical/hybrid/composition_common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace hbfsim::physical::hybrid {
namespace {

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs + rhs;
}

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs * rhs;
}

} // namespace

const char* to_string(BackingTier tier) {
    switch (tier) {
    case BackingTier::Hbf:
        return "hbf";
    case BackingTier::External:
        return "external";
    }
    throw std::runtime_error("unknown backing tier");
}

std::string to_string(SemanticKind kind) {
    switch (kind) {
    case SemanticKind::Unknown:
        return "unknown";
    case SemanticKind::ModelWeights:
        return "model_weights";
    case SemanticKind::SharedContext:
        return "shared_context";
    case SemanticKind::GeneratedContext:
        return "generated_context";
    case SemanticKind::Scratch:
        return "scratch";
    case SemanticKind::Metadata:
        return "metadata";
    }
    throw std::runtime_error("unknown semantic kind");
}

void validate_memory_requests(
    const std::vector<MemoryRequest>& requests,
    std::string_view context) {
    std::optional<double> previous_arrival;
    std::optional<std::uint64_t> previous_phase;
    const bool has_explicit_phases = std::any_of(
        requests.begin(),
        requests.end(),
        [](const MemoryRequest& request) { return request.phase.has_value(); });
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        const auto prefix = std::string(context) + " request " + std::to_string(i);
        if (request.bytes == 0) {
            throw std::runtime_error(prefix + " bytes must be positive");
        }
        if (!std::isfinite(request.arrival_ns) || request.arrival_ns < 0.0) {
            throw std::runtime_error(
                prefix + " arrival must be finite and non-negative");
        }
        if (previous_arrival && request.arrival_ns < *previous_arrival) {
            throw std::runtime_error(
                std::string(context) +
                " arrivals must be nondecreasing; sort the trace before simulation");
        }
        (void)checked_add(
            request.addr,
            request.bytes - 1,
            "memory request end address");
        if (has_explicit_phases && !request.phase) {
            throw std::runtime_error(
                std::string(context) +
                " must specify phase= on every request once phase scheduling is used");
        }
        if (request.phase) {
            if (previous_phase && *request.phase < *previous_phase) {
                throw std::runtime_error(
                    std::string(context) +
                    " phases must be nondecreasing");
            }
            previous_phase = request.phase;
        }
        if (request.compute_ns) {
            if (!request.layer) {
                throw std::runtime_error(
                    prefix + " compute_ns requires an explicit layer=");
            }
            if (!std::isfinite(*request.compute_ns) ||
                *request.compute_ns < 0.0) {
                throw std::runtime_error(
                    prefix + " compute_ns must be finite and non-negative");
            }
        }
        previous_arrival = request.arrival_ns;
    }
}

AddressHeatmapConfig make_composition_address_heatmap_config(
    const hbm::HbmConfig& hbm_config,
    const hbf::HbfConfig& hbf_config,
    const std::vector<MemoryRequest>& requests,
    std::size_t bin_count,
    std::uint64_t external_capacity_bytes) {
    if (bin_count == 0) {
        throw std::runtime_error(
            "enabled composition address heatmap requires at least one bin");
    }
    auto hbf_pages = static_cast<std::uint64_t>(hbf_config.stacks);
    hbf_pages = checked_mul(
        hbf_pages, hbf_config.channels_per_stack, "HBF heatmap capacity");
    hbf_pages = checked_mul(
        hbf_pages, hbf_config.dies_per_channel, "HBF heatmap capacity");
    hbf_pages = checked_mul(
        hbf_pages, hbf_config.planes_per_die, "HBF heatmap capacity");
    hbf_pages = checked_mul(
        hbf_pages, hbf_config.blocks_per_plane, "HBF heatmap capacity");
    hbf_pages = checked_mul(
        hbf_pages, hbf_config.pages_per_block, "HBF heatmap capacity");
    const auto hbf_capacity_bytes = checked_mul(
        hbf_pages, hbf_config.page_size_bytes, "HBF heatmap byte capacity");

    // Keep the workload/HBF-logical view comparable across policies. The
    // external device has its own independently sized physical domain below;
    // observed requests still extend the logical domain when needed.
    AddressBoundary logical_extent{
        std::max(hbm_config.capacity_bytes, hbf_capacity_bytes)};
    struct SemanticExtent {
        bool present = false;
        std::uint64_t begin = 0;
        AddressBoundary end = 0;
    };
    std::array<SemanticExtent, 6> semantic_extents{};
    for (const auto& request : requests) {
        const auto end = address_exclusive_end(request.addr, request.bytes);
        logical_extent = std::max(logical_extent, end);
        const auto kind = static_cast<std::size_t>(request.kind);
        if (kind >= semantic_extents.size() ||
            request.kind == SemanticKind::Unknown) {
            continue;
        }
        auto& extent = semantic_extents[kind];
        if (!extent.present) {
            extent = SemanticExtent{
                .present = true,
                .begin = request.addr,
                .end = end,
            };
        } else {
            extent.begin = std::min(extent.begin, request.addr);
            extent.end = std::max(extent.end, end);
        }
    }
    const AddressBoundary minimum_size{static_cast<std::uint64_t>(bin_count)};
    logical_extent = std::max(logical_extent, minimum_size);

    AddressHeatmapConfig config;
    config.bin_count = bin_count;
    config.domains[static_cast<std::size_t>(AddressDomain::WorkloadLogical)].size_bytes =
        logical_extent;
    config.domains[static_cast<std::size_t>(AddressDomain::HbmPhysical)].size_bytes =
        std::max(AddressBoundary{hbm_config.capacity_bytes}, minimum_size);
    config.domains[static_cast<std::size_t>(AddressDomain::HbfLogical)].size_bytes =
        logical_extent;
    config.domains[static_cast<std::size_t>(AddressDomain::HbfPhysical)].size_bytes =
        std::max(AddressBoundary{hbf_capacity_bytes}, minimum_size);
    config.domains[
        static_cast<std::size_t>(AddressDomain::ExternalPhysical)].size_bytes =
        std::max(AddressBoundary{external_capacity_bytes}, minimum_size);

    const auto add_semantic_regions = [&](AddressDomain domain) {
        auto& regions = config.domains[static_cast<std::size_t>(domain)].regions;
        const auto add = [&](SemanticKind kind, AddressRegionKind region_kind) {
            const auto& extent = semantic_extents[static_cast<std::size_t>(kind)];
            if (!extent.present) {
                return;
            }
            regions.push_back(AddressRegion{
                // This is deliberately named an observed extent: requests of
                // one semantic kind can be disjoint, so min/max is a visual
                // bounding box, not a claim that every intervening byte is
                // owned by that kind. Traffic bins remain exact.
                .name = "semantic_observed_extent/" + to_string(kind),
                .kind = region_kind,
                .begin = extent.begin,
                .end = extent.end,
            });
        };
        add(SemanticKind::ModelWeights, AddressRegionKind::StaticData);
        add(SemanticKind::SharedContext, AddressRegionKind::StaticData);
        add(SemanticKind::GeneratedContext, AddressRegionKind::MappedData);
        add(SemanticKind::Scratch, AddressRegionKind::Workload);
        add(SemanticKind::Metadata, AddressRegionKind::Reserved);
    };
    add_semantic_regions(AddressDomain::WorkloadLogical);
    add_semantic_regions(AddressDomain::HbfLogical);
    return config;
}

void record_workload_address_traffic(
    AddressHeatmap& heatmap,
    const std::vector<MemoryRequest>& requests) {
    for (const auto& request : requests) {
        heatmap.record(AddressTrafficRecord{
            .domain = AddressDomain::WorkloadLogical,
            .direction = request.op == Op::Read ?
                TrafficDirection::Read : TrafficDirection::Write,
            .source = HeatmapTrafficSource::Workload,
            .address = request.addr,
            .bytes = request.bytes,
        });
    }
}

std::uint64_t hbf_page_capacity(const hbf::HbfDevice& device) {
    const auto& cfg = device.config();
    auto pages = static_cast<std::uint64_t>(cfg.stacks);
    pages = checked_mul(pages, cfg.channels_per_stack, "HBF page capacity");
    pages = checked_mul(pages, cfg.dies_per_channel, "HBF page capacity");
    pages = checked_mul(pages, cfg.planes_per_die, "HBF page capacity");
    pages = checked_mul(pages, cfg.blocks_per_plane, "HBF page capacity");
    return checked_mul(pages, cfg.pages_per_block, "HBF page capacity");
}

std::uint64_t map_static_hbf_page_addr(
    const hbf::HbfDevice& device,
    std::uint64_t source_page) {
    const auto& cfg = device.config();
    const auto capacity = hbf_page_capacity(device);
    (void)checked_mul(capacity, cfg.page_size_bytes, "HBF byte-address capacity");
    if (source_page >= capacity) {
        throw std::runtime_error(
            "static HBF source page " + std::to_string(source_page) +
            " exceeds physical page capacity " + std::to_string(capacity));
    }

    auto planes_per_stack = static_cast<std::uint64_t>(cfg.planes_per_die);
    planes_per_stack = checked_mul(
        planes_per_stack, cfg.dies_per_channel, "static HBF planes per stack");
    planes_per_stack = checked_mul(
        planes_per_stack, cfg.channels_per_stack, "static HBF planes per stack");

    // Static-direct access is the FTL-bypass counterpart of managed FLAT
    // routing, not a second placement rule. Reuse the managed FTL's
    // logical-page stack owner and its per-stack round-robin data-plane order.
    // The resulting mapping is a full-capacity bijection but removes
    // translation and mapping pages entirely.
    const auto stack = device.stack_for_logical_page(source_page);
    const auto stack_page_index = source_page / cfg.stacks;
    auto local_plane = stack_page_index % planes_per_stack;
    const auto page_in_plane = stack_page_index / planes_per_stack;

    hbf::HbfAddress addr;
    addr.plane = static_cast<std::uint32_t>(
        local_plane % cfg.planes_per_die);
    local_plane /= cfg.planes_per_die;
    addr.die = static_cast<std::uint32_t>(
        local_plane % cfg.dies_per_channel);
    local_plane /= cfg.dies_per_channel;
    addr.channel = static_cast<std::uint32_t>(
        local_plane % cfg.channels_per_stack);
    addr.stack = static_cast<std::uint32_t>(stack);
    addr.page = static_cast<std::uint32_t>(
        page_in_plane % cfg.pages_per_block);
    addr.block = static_cast<std::uint32_t>(
        page_in_plane / cfg.pages_per_block);
    addr.offset = 0;
    return device.encode(addr);
}

std::uint64_t unmap_static_hbf_page_addr(
    const hbf::HbfDevice& device,
    std::uint64_t physical_page_addr) {
    const auto& cfg = device.config();
    if (physical_page_addr % cfg.page_size_bytes != 0) {
        throw std::runtime_error("static HBF physical page address must be page-aligned");
    }
    const auto capacity = hbf_page_capacity(device);
    const auto physical_page = physical_page_addr / cfg.page_size_bytes;
    if (physical_page >= capacity) {
        throw std::runtime_error(
            "static HBF physical page " + std::to_string(physical_page) +
            " exceeds physical page capacity " + std::to_string(capacity));
    }
    const auto addr = device.decode(physical_page_addr);
    auto planes_per_stack = static_cast<std::uint64_t>(cfg.planes_per_die);
    planes_per_stack = checked_mul(
        planes_per_stack,
        cfg.dies_per_channel,
        "static HBF inverse planes per stack");
    planes_per_stack = checked_mul(
        planes_per_stack,
        cfg.channels_per_stack,
        "static HBF inverse planes per stack");
    auto local_plane = static_cast<std::uint64_t>(addr.channel);
    local_plane = checked_add(
        checked_mul(
            local_plane,
            cfg.dies_per_channel,
            "static HBF inverse local plane"),
        addr.die,
        "static HBF inverse local plane");
    local_plane = checked_add(
        checked_mul(
            local_plane,
            cfg.planes_per_die,
            "static HBF inverse local plane"),
        addr.plane,
        "static HBF inverse local plane");
    const auto page_in_plane = checked_add(
        checked_mul(
            addr.block,
            cfg.pages_per_block,
            "static HBF inverse page in plane"),
        addr.page,
        "static HBF inverse page in plane");
    const auto stack_page_index = checked_add(
        checked_mul(
            page_in_plane,
            planes_per_stack,
            "static HBF inverse stack page"),
        local_plane,
        "static HBF inverse stack page");
    const auto stripe_base = checked_mul(
        stack_page_index,
        cfg.stacks,
        "static HBF inverse stripe base");
    for (std::uint64_t lane = 0; lane < cfg.stacks; ++lane) {
        const auto candidate = checked_add(
            stripe_base,
            lane,
            "static HBF inverse source page");
        if (device.stack_for_logical_page(candidate) == addr.stack) {
            return candidate;
        }
    }
    throw std::runtime_error(
        "static HBF inverse could not recover the logical stack lane");
}

std::uint64_t static_hbf_addr_for_byte(
    const hbf::HbfDevice& device,
    std::uint64_t source_byte_addr) {
    const auto page_size = device.config().page_size_bytes;
    const auto page = source_byte_addr / page_size;
    const auto offset = source_byte_addr % page_size;
    return checked_add(
        map_static_hbf_page_addr(device, page),
        offset,
        "static HBF byte address");
}

std::vector<std::uint64_t> collect_initial_read_lpns(
    const std::vector<MemoryRequest>& requests,
    std::uint64_t page_size,
    std::optional<std::uint64_t> min_addr) {
    if (page_size == 0) {
        throw std::runtime_error("initial-image page size must be positive");
    }
    struct WrittenRange {
        std::uint64_t begin = 0;
        std::uint64_t end = 0;
    };
    std::map<std::uint64_t, std::vector<WrittenRange>> written_ranges;
    std::set<std::uint64_t> lpns;
    for (const auto& request : requests) {
        if (request.bytes == 0) {
            throw std::runtime_error("initial-image request bytes must be positive");
        }
        auto cursor = request.addr;
        auto remaining = request.bytes;
        while (remaining != 0) {
            const auto lpn = cursor / page_size;
            const auto begin = cursor % page_size;
            const auto bytes = std::min(remaining, page_size - begin);
            const auto end = checked_add(begin, bytes, "initial-image page range");
            if (request.op == Op::Read) {
                bool covered = false;
                const auto found = written_ranges.find(lpn);
                if (found != written_ranges.end()) {
                    auto covered_through = begin;
                    for (const auto& range : found->second) {
                        if (range.end <= covered_through) {
                            continue;
                        }
                        if (range.begin > covered_through) {
                            break;
                        }
                        covered_through = std::max(covered_through, range.end);
                        if (covered_through >= end) {
                            covered = true;
                            break;
                        }
                    }
                }
                const auto segment_last = checked_add(
                    cursor,
                    bytes - 1,
                    "initial-image read segment end");
                if (!covered && (!min_addr || segment_last >= *min_addr)) {
                    lpns.insert(lpn);
                }
            } else if (request.op == Op::Write) {
                auto& ranges = written_ranges[lpn];
                ranges.push_back(WrittenRange{.begin = begin, .end = end});
                std::sort(
                    ranges.begin(),
                    ranges.end(),
                    [](const WrittenRange& lhs, const WrittenRange& rhs) {
                        return lhs.begin < rhs.begin ||
                            (lhs.begin == rhs.begin && lhs.end < rhs.end);
                    });
                std::vector<WrittenRange> merged;
                for (const auto& range : ranges) {
                    if (merged.empty() || range.begin > merged.back().end) {
                        merged.push_back(range);
                    } else {
                        merged.back().end = std::max(merged.back().end, range.end);
                    }
                }
                ranges = std::move(merged);
            }
            remaining -= bytes;
            if (remaining != 0) {
                cursor = checked_add(cursor, bytes, "next initial-image segment");
            }
        }
    }
    return {lpns.begin(), lpns.end()};
}

double BaseDieLinkStats::active_span_ns() const {
    return std::isfinite(first_arrival_ns) && finish_ns > first_arrival_ns ?
        finish_ns - first_arrival_ns : 0.0;
}

double BaseDieLinkStats::read_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || links == 0 ? 0.0 :
        read_busy_ns / (span * static_cast<double>(links));
}

double BaseDieLinkStats::write_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || links == 0 ? 0.0 :
        write_busy_ns / (span * static_cast<double>(links));
}

BaseDieLink::BaseDieLink(BaseDieLinkConfig config, std::string entity)
    : config_(config), entity_(std::move(entity)) {
    if (!(config_.read_bandwidth_GBps > 0.0) ||
        !std::isfinite(config_.read_bandwidth_GBps) ||
        !(config_.write_bandwidth_GBps > 0.0) ||
        !std::isfinite(config_.write_bandwidth_GBps) ||
        config_.latency_ns < 0.0 || !std::isfinite(config_.latency_ns)) {
        throw std::runtime_error(
            "base-die link bandwidth must be positive and finite; latency must "
            "be non-negative and finite");
    }
}

PhysicalCompletion BaseDieLink::issue(
    std::string id,
    Op op,
    double arrival_ns,
    std::uint64_t bytes,
    TraceConfig trace) {
    if (op != Op::Read && op != Op::Write) {
        throw std::runtime_error("base-die link only supports read/write transfers");
    }
    if (!std::isfinite(arrival_ns) || arrival_ns < 0.0) {
        throw std::runtime_error("base-die link arrival must be finite and non-negative");
    }
    if (bytes == 0) {
        throw std::runtime_error("base-die link transfer bytes must be positive");
    }
    stats_.first_arrival_ns = std::min(stats_.first_arrival_ns, arrival_ns);
    const bool is_read = op == Op::Read;
    auto& ready = is_read ? read_ready_ns_ : write_ready_ns_;
    auto& queue_wait = is_read ? stats_.read_queue_wait_ns : stats_.write_queue_wait_ns;
    auto& busy = is_read ? stats_.read_busy_ns : stats_.write_busy_ns;
    auto& fixed_latency_work = is_read ?
        stats_.read_fixed_latency_work_ns :
        stats_.write_fixed_latency_work_ns;
    auto& transfer_count = is_read ? stats_.read_transfers : stats_.write_transfers;
    auto& byte_count = is_read ? stats_.read_bytes : stats_.write_bytes;
    const double bandwidth = is_read ? config_.read_bandwidth_GBps : config_.write_bandwidth_GBps;
    const double serialization_ns = transfer_time_ns(bytes, bandwidth);
    const double start_ns = std::max(arrival_ns, ready);
    const double finish_ns = start_ns + serialization_ns + config_.latency_ns;
    if (!std::isfinite(serialization_ns) || !std::isfinite(finish_ns)) {
        throw std::runtime_error("base-die link timing exceeds finite double range");
    }
    const double wait_ns = start_ns - arrival_ns;
    ready = start_ns + serialization_ns;

    queue_wait += wait_ns;
    busy += serialization_ns;
    fixed_latency_work += config_.latency_ns;
    transfer_count = checked_add(
        transfer_count,
        1,
        "base-die link transfer count");
    byte_count = checked_add(
        byte_count,
        bytes,
        "base-die link byte count");
    stats_.finish_ns = std::max(stats_.finish_ns, finish_ns);

    PhysicalCompletion out;
    out.id = std::move(id);
    out.tier = Tier::HBF;
    out.op = op;
    out.arrival_ns = arrival_ns;
    out.start_ns = start_ns;
    out.finish_ns = finish_ns;
    out.logical_bytes = bytes;
    out.physical_bytes = bytes;
    out.resource_path = entity_;
    out.note = is_read ? "hbf-to-hbm-staging-link" : "hbm-to-hbf-backing-link";
    out.breakdown.scheduler_queue_wait_ns = wait_ns;
    out.breakdown.hb_io_transfer_ns = serialization_ns;
    out.breakdown.command_ns = config_.latency_ns;
    auto* spans = trace_spans_enabled(trace) ? &out.spans : nullptr;
    add_trace_span(
        spans,
        is_read ? "base_die_link_read_serialize" : "base_die_link_write_serialize",
        "base_die_link",
        entity_,
        start_ns,
        start_ns + serialization_ns,
        true,
        std::to_string(bytes) + "B");
    add_trace_span(
        spans,
        is_read ? "base_die_link_read_latency" : "base_die_link_write_latency",
        "base_die_link",
        entity_,
        start_ns + serialization_ns,
        finish_ns,
        true,
        std::to_string(bytes) + "B");
    return out;
}

void record_user_op(
    CompositionRunResult& result,
    const MemoryRequest& request,
    double finish_ns,
    double service_start_ns,
    double offered_arrival_ns,
    double source_arrival_ns) {
    if (!std::isfinite(finish_ns) ||
        !std::isfinite(service_start_ns) ||
        !std::isfinite(offered_arrival_ns) ||
        !std::isfinite(source_arrival_ns) ||
        finish_ns < service_start_ns ||
        service_start_ns < offered_arrival_ns ||
        offered_arrival_ns < source_arrival_ns) {
        throw std::runtime_error(
            "composition user latency frontiers violate causal order");
    }
    result.ops = checked_add(result.ops, 1, "composition operation count");
    result.reads = checked_add(
        result.reads,
        request.op == Op::Read ? 1 : 0,
        "composition read count");
    result.writes = checked_add(
        result.writes,
        request.op == Op::Write ? 1 : 0,
        "composition write count");
    result.logical_bytes = checked_add(
        result.logical_bytes,
        request.bytes,
        "composition logical byte count");
    result.finish_ns = std::max(result.finish_ns, finish_ns);
    result.user_finish_ns = std::max(result.user_finish_ns, finish_ns);
    result.service_latencies_ns.push_back(finish_ns - service_start_ns);
    result.offered_latencies_ns.push_back(finish_ns - offered_arrival_ns);
    result.source_latencies_ns.push_back(finish_ns - source_arrival_ns);
}

void add_user_completion(
    CompositionRunResult& result,
    PhysicalCompletion completion,
    bool hbm,
    bool hbf) {
    result.finish_ns = std::max(result.finish_ns, completion.finish_ns);
    if (hbm) {
        result.hbm_user_accesses++;
    }
    if (hbf) {
        result.hbf_user_accesses++;
    }
    if (completion.finish_ns < completion.arrival_ns) {
        result.warnings.push_back(completion.id + " finished before arrival");
    }
    if (completion.note.find("unmapped") != std::string::npos) {
        result.warnings.push_back(completion.id + " produced " + completion.note);
    }
    if (result.retain_completions) {
        result.completions.push_back(std::move(completion));
    }
}

void add_background_completion(
    CompositionRunResult& result,
    PhysicalCompletion completion) {
    result.finish_ns = std::max(result.finish_ns, completion.finish_ns);
    if (completion.finish_ns < completion.arrival_ns) {
        result.warnings.push_back(completion.id + " finished before arrival");
    }
    if (result.retain_completions) {
        result.completions.push_back(std::move(completion));
    }
}

} // namespace hbfsim::physical::hybrid
