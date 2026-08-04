#include "physical/hybrid/capacity_overflow_composition.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace hbfsim::physical::hybrid {
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

std::uint64_t checked_mul(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* name) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs * rhs;
}

void require_finite_nonnegative(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error(
            std::string(name) + " must be finite and non-negative");
    }
}

PhysicalRequest make_request(
    std::string id,
    Tier tier,
    Op op,
    double arrival_ns,
    std::uint64_t addr,
    std::uint64_t bytes,
    AddressSpace address_space,
    const TraceConfig& trace,
    HeatmapTrafficSource source) {
    return PhysicalRequest{
        .id = std::move(id),
        .tier = tier,
        .op = op,
        .address_space = address_space,
        .trace = trace,
        .arrival_ns = arrival_ns,
        .addr = addr,
        .bytes = bytes,
        .stream_id = 0,
        .heatmap_source = source,
    };
}

struct TransferItem {
    std::uint64_t ordinal = 0;
    std::uint64_t slot = 0;
    double ready_ns = 0.0;
};

void sort_by_ready(std::vector<TransferItem>& items) {
    std::stable_sort(
        items.begin(),
        items.end(),
        [](const TransferItem& lhs, const TransferItem& rhs) {
            if (lhs.ready_ns != rhs.ready_ns) {
                return lhs.ready_ns < rhs.ready_ns;
            }
            return lhs.ordinal < rhs.ordinal;
        });
}

BaseDieLinkStats aggregate_link_stats(
    const std::vector<BaseDieLink>& links) {
    BaseDieLinkStats total;
    total.links = links.size();
    for (const auto& link : links) {
        const auto& stats = link.stats();
        total.read_transfers = checked_add(
            total.read_transfers,
            stats.read_transfers,
            "capacity-overflow D2D read transfers");
        total.write_transfers = checked_add(
            total.write_transfers,
            stats.write_transfers,
            "capacity-overflow D2D write transfers");
        total.read_bytes = checked_add(
            total.read_bytes,
            stats.read_bytes,
            "capacity-overflow D2D read bytes");
        total.write_bytes = checked_add(
            total.write_bytes,
            stats.write_bytes,
            "capacity-overflow D2D write bytes");
        total.read_queue_wait_ns += stats.read_queue_wait_ns;
        total.write_queue_wait_ns += stats.write_queue_wait_ns;
        total.read_busy_ns += stats.read_busy_ns;
        total.write_busy_ns += stats.write_busy_ns;
        total.read_fixed_latency_work_ns +=
            stats.read_fixed_latency_work_ns;
        total.write_fixed_latency_work_ns +=
            stats.write_fixed_latency_work_ns;
        total.first_arrival_ns = std::min(
            total.first_arrival_ns,
            stats.first_arrival_ns);
        total.finish_ns = std::max(total.finish_ns, stats.finish_ns);
    }
    return total;
}

} // namespace

const char* to_string(CapacityReadbackDestination destination) {
    switch (destination) {
    case CapacityReadbackDestination::DmaBuffer:
        return "dma-buffer";
    case CapacityReadbackDestination::OriginalSlots:
        return "original-hbm-slots";
    }
    throw std::runtime_error(
        "unknown capacity-overflow readback destination");
}

CapacityOverflowComposition::CapacityOverflowComposition(
    CapacityOverflowConfig config)
    : config_(std::move(config)) {}

CapacityOverflowRunResult CapacityOverflowComposition::run(
    const CapacityOverflowWorkload& workload) {
    require_finite_nonnegative(
        workload.first_arrival_ns,
        "capacity-overflow first arrival");
    require_finite_nonnegative(
        workload.interarrival_ns,
        "capacity-overflow interarrival");
    if (workload.total_write_pages == 0) {
        throw std::runtime_error(
            "capacity-overflow total write pages must be positive");
    }
    if (config_.address_heatmap_bins == 0 ||
        config_.address_heatmap_bins > kMaxAddressHeatmapBins) {
        throw std::runtime_error(
            "capacity-overflow heatmap bins must be in [1, 8192]");
    }
    if (config_.transfer_batch_pages == 0) {
        throw std::runtime_error(
            "capacity-overflow transfer batch must be positive");
    }

    const auto page_size = config_.backing == BackingTier::Hbf ?
        config_.hbf.page_size_bytes :
        config_.external_backing.page_size_bytes;
    if (page_size == 0 ||
        workload.base_addr % page_size != 0 ||
        config_.hbm.capacity_bytes % page_size != 0 ||
        config_.read_buffer_bytes == 0 ||
        config_.read_buffer_bytes % page_size != 0 ||
        config_.read_buffer_bytes >= config_.hbm.capacity_bytes) {
        throw std::runtime_error(
            "capacity-overflow HBM/read-buffer geometry must be positive, "
            "page aligned, and leave a non-empty resident window");
    }
    if (config_.backing == BackingTier::Hbf) {
        if (config_.external_backing.page_size_bytes != page_size &&
            config_.external_backing.page_size_bytes != 0) {
            // The unused profile may differ, but a non-default mismatch here
            // often signals that the caller accidentally compared unequal
            // granularities. Fail closed instead of silently ignoring it.
            throw std::runtime_error(
                "capacity-overflow configured page sizes must agree");
        }
        if (!config_.hbf.write_buffer_completion_requires_flush) {
            throw std::runtime_error(
                "capacity-overflow HBF comparison requires payload-page "
                "program completion before an HBM victim slot is reused");
        }
    } else if (config_.hbf.page_size_bytes != page_size) {
        throw std::runtime_error(
            "capacity-overflow configured page sizes must agree");
    }

    const auto data_capacity_bytes =
        config_.hbm.capacity_bytes - config_.read_buffer_bytes;
    const auto data_pages = data_capacity_bytes / page_size;
    const auto read_buffer_pages = config_.read_buffer_bytes / page_size;
    if (data_pages == 0 || read_buffer_pages == 0 ||
        config_.transfer_batch_pages > data_pages ||
        config_.transfer_batch_pages > read_buffer_pages) {
        throw std::runtime_error(
            "capacity-overflow transfer batch must fit both HBM regions");
    }
    if (workload.total_write_pages <= data_pages) {
        throw std::runtime_error(
            "capacity-overflow workload must exceed the HBM resident window");
    }
    const auto offload_pages =
        workload.total_write_pages - data_pages;
    if (config_.readback_destination ==
            CapacityReadbackDestination::OriginalSlots &&
        offload_pages > data_pages) {
        throw std::runtime_error(
            "capacity-overflow original-slot readback requires the "
            "offloaded set to fit the HBM resident window");
    }
    const auto total_write_bytes = checked_mul(
        workload.total_write_pages,
        page_size,
        "capacity-overflow total write bytes");
    (void)checked_add(
        workload.base_addr,
        total_write_bytes,
        "capacity-overflow workload end");
    const auto offload_bytes = checked_mul(
        offload_pages,
        page_size,
        "capacity-overflow offloading bytes");
    const auto offload_end = checked_add(
        workload.base_addr,
        offload_bytes,
        "capacity-overflow backing end");
    const auto last_arrival_ns = workload.first_arrival_ns +
        static_cast<double>(workload.total_write_pages - 1) *
            workload.interarrival_ns;
    if (!std::isfinite(last_arrival_ns)) {
        throw std::runtime_error(
            "capacity-overflow last write arrival is not finite");
    }

    std::vector<MemoryRequest> footprint{
        MemoryRequest{
            .id = "capacity-overflow-write-footprint",
            .op = Op::Write,
            .addr = workload.base_addr,
            .bytes = total_write_bytes,
            .arrival_ns = workload.first_arrival_ns,
            .index = 0,
            .kind = SemanticKind::GeneratedContext,
            .label = "kv-append",
        },
        MemoryRequest{
            .id = "capacity-overflow-readback-footprint",
            .op = Op::Read,
            .addr = workload.base_addr,
            .bytes = offload_bytes,
            .arrival_ns = last_arrival_ns,
            .index = 1,
            .kind = SemanticKind::GeneratedContext,
            .label = "kv-readback",
        },
    };
    auto heatmap_config = make_composition_address_heatmap_config(
        config_.hbm,
        config_.hbf,
        footprint,
        config_.address_heatmap_bins,
        config_.backing == BackingTier::External ?
            config_.external_backing.capacity_bytes : 0);
    auto& hbm_domain = heatmap_config.domains[
        static_cast<std::size_t>(AddressDomain::HbmPhysical)];
    hbm_domain.regions.push_back(AddressRegion{
        .name = "capacity_overflow_resident_window",
        .kind = AddressRegionKind::MappedData,
        .begin = 0,
        .end = data_capacity_bytes,
    });
    hbm_domain.regions.push_back(AddressRegion{
        .name = "capacity_overflow_read_buffer",
        .kind = AddressRegionKind::CooperativeBuffer,
        .begin = data_capacity_bytes,
        .end = config_.hbm.capacity_bytes,
    });
    AddressHeatmap heatmap(std::move(heatmap_config));
    heatmap.record_contiguous_accesses(
        AddressTrafficRecord{
            .domain = AddressDomain::WorkloadLogical,
            .direction = TrafficDirection::Write,
            .source = HeatmapTrafficSource::Workload,
            .address = workload.base_addr,
            .bytes = page_size,
        },
        workload.total_write_pages);
    heatmap.record_contiguous_accesses(
        AddressTrafficRecord{
            .domain = AddressDomain::WorkloadLogical,
            .direction = TrafficDirection::Read,
            .source = HeatmapTrafficSource::Workload,
            .address = workload.base_addr,
            .bytes = page_size,
        },
        offload_pages);

    hbm::HbmDevice hbm(config_.hbm, &heatmap);
    std::optional<hbf::HbfDevice> hbf;
    std::optional<external::ExternalBackingDevice> external;
    std::vector<BaseDieLink> links;
    if (config_.backing == BackingTier::Hbf) {
        hbf.emplace(config_.hbf, &heatmap);
        const auto backing_capacity = checked_mul(
            hbf_page_capacity(*hbf),
            page_size,
            "capacity-overflow HBF capacity");
        if (offload_end > backing_capacity) {
            throw std::runtime_error(
                "capacity-overflow offloaded address range exceeds HBF capacity");
        }
        links.reserve(config_.hbf.stacks);
        for (std::size_t stack = 0; stack < config_.hbf.stacks; ++stack) {
            links.emplace_back(
                config_.base_die_link,
                "capacity_overflow/base_die_link/stack" +
                    std::to_string(stack));
        }
    } else {
        external.emplace(config_.external_backing, &heatmap);
        if (offload_end > config_.external_backing.capacity_bytes) {
            throw std::runtime_error(
                "capacity-overflow offloaded address range exceeds external capacity");
        }
    }

    CapacityOverflowRunResult result;
    auto& stats = result.stats;
    stats.backing = config_.backing;
    stats.readback_destination = config_.readback_destination;
    stats.page_size_bytes = page_size;
    stats.hbm_capacity_bytes = config_.hbm.capacity_bytes;
    stats.hbm_data_capacity_bytes = data_capacity_bytes;
    stats.hbm_read_buffer_bytes = config_.read_buffer_bytes;
    stats.hbm_data_pages = data_pages;
    stats.hbm_read_buffer_pages = read_buffer_pages;
    stats.transfer_batch_pages = config_.transfer_batch_pages;
    stats.written_pages = workload.total_write_pages;
    stats.offload_pages = offload_pages;
    stats.readback_pages = offload_pages;
    stats.offload_bytes = offload_bytes;
    stats.readback_bytes = offload_bytes;
    stats.first_arrival_ns = workload.first_arrival_ns;
    result.fill_write_latencies_ns.reserve(data_pages);
    result.offload_offered_latencies_ns.reserve(offload_pages);
    result.offload_service_latencies_ns.reserve(offload_pages);
    result.readback_offered_latencies_ns.reserve(offload_pages);
    result.readback_service_latencies_ns.reserve(offload_pages);

    const auto absorb = [&](PhysicalCompletion completion) {
        stats.quiescent_finish_ns = std::max(
            stats.quiescent_finish_ns,
            completion.finish_ns);
        if (config_.retain_completions) {
            result.completions.push_back(std::move(completion));
        }
    };

    const auto logical_addr = [&](std::uint64_t ordinal) {
        return checked_add(
            workload.base_addr,
            checked_mul(
                ordinal,
                page_size,
                "capacity-overflow logical page offset"),
            "capacity-overflow logical page address");
    };
    const auto source_arrival = [&](std::uint64_t ordinal) {
        const auto arrival = workload.first_arrival_ns +
            static_cast<double>(ordinal) * workload.interarrival_ns;
        if (!std::isfinite(arrival)) {
            throw std::runtime_error(
                "capacity-overflow page arrival is not finite");
        }
        return arrival;
    };

    // Phase 1a: fill every resident data slot exactly once.
    std::vector<std::uint64_t> resident_ordinal(data_pages);
    std::iota(
        resident_ordinal.begin(),
        resident_ordinal.end(),
        std::uint64_t{0});
    double fill_finish_ns = workload.first_arrival_ns;
    for (std::uint64_t ordinal = 0; ordinal < data_pages; ++ordinal) {
        const auto arrival_ns = source_arrival(ordinal);
        auto completion = hbm.issue(make_request(
            "fill/page" + std::to_string(ordinal),
            Tier::HBM,
            Op::Write,
            arrival_ns,
            checked_mul(
                ordinal,
                page_size,
                "capacity-overflow fill HBM address"),
            page_size,
            AddressSpace::Physical,
            config_.trace,
            HeatmapTrafficSource::Workload));
        fill_finish_ns = std::max(fill_finish_ns, completion.finish_ns);
        result.fill_write_latencies_ns.push_back(
            completion.finish_ns - arrival_ns);
        ++result.hbm_user_accesses;
        absorb(std::move(completion));
    }
    stats.fill_finish_ns = fill_finish_ns;

    // Phase 1b: evict FIFO victims in bounded batches. A batch may use many
    // independent slots, but a slot is never overwritten before its backing
    // write has completed.
    double prior_batch_finish_ns = fill_finish_ns;
    for (std::uint64_t batch_begin = 0;
         batch_begin < offload_pages;
         batch_begin += config_.transfer_batch_pages) {
        const auto batch_count = std::min<std::uint64_t>(
            config_.transfer_batch_pages,
            offload_pages - batch_begin);
        const auto newest_new_ordinal =
            data_pages + batch_begin + batch_count - 1;
        const auto batch_start_ns = std::max(
            prior_batch_finish_ns,
            source_arrival(newest_new_ordinal));
        if (batch_begin == 0) {
            stats.offload_start_ns = batch_start_ns;
        }

        std::vector<TransferItem> items;
        items.reserve(batch_count);
        for (std::uint64_t offset = 0; offset < batch_count; ++offset) {
            const auto offload_ordinal = batch_begin + offset;
            const auto slot = offload_ordinal % data_pages;
            if (resident_ordinal.at(slot) != offload_ordinal) {
                throw std::runtime_error(
                    "capacity-overflow FIFO victim order is inconsistent");
            }
            auto completion = hbm.issue(make_request(
                "offload/victim" + std::to_string(offload_ordinal) +
                    "/hbm-read",
                Tier::HBM,
                Op::Read,
                batch_start_ns,
                checked_mul(
                    slot,
                    page_size,
                    "capacity-overflow victim HBM address"),
                page_size,
                AddressSpace::Physical,
                config_.trace,
                HeatmapTrafficSource::Destage));
            items.push_back(TransferItem{
                .ordinal = offload_ordinal,
                .slot = slot,
                .ready_ns = completion.finish_ns,
            });
            ++result.hbm_background_accesses;
            absorb(std::move(completion));
        }

        sort_by_ready(items);
        if (hbf) {
            for (auto& item : items) {
                const auto lpn =
                    logical_addr(item.ordinal) / page_size;
                const auto stack = hbf->stack_for_logical_page(lpn);
                auto completion = links.at(stack).issue(
                    "offload/victim" + std::to_string(item.ordinal) +
                        "/d2d-write",
                    Op::Write,
                    item.ready_ns,
                    page_size,
                    config_.trace);
                item.ready_ns = completion.finish_ns;
                absorb(std::move(completion));
            }
            sort_by_ready(items);
            for (auto& item : items) {
                auto completion = hbf->issue(make_request(
                    "offload/victim" + std::to_string(item.ordinal) +
                        "/hbf-write",
                    Tier::HBF,
                    Op::Write,
                    item.ready_ns,
                    logical_addr(item.ordinal),
                    page_size,
                    AddressSpace::Logical,
                    config_.trace,
                    HeatmapTrafficSource::Destage));
                item.ready_ns = completion.finish_ns;
                ++result.hbf_background_accesses;
                absorb(std::move(completion));
            }
        } else {
            for (auto& item : items) {
                auto completion = external->issue(make_request(
                    "offload/victim" + std::to_string(item.ordinal) +
                        "/external-write",
                    Tier::External,
                    Op::Write,
                    item.ready_ns,
                    logical_addr(item.ordinal),
                    page_size,
                    AddressSpace::Logical,
                    config_.trace,
                    HeatmapTrafficSource::Destage));
                item.ready_ns = completion.finish_ns;
                ++result.external_background_accesses;
                absorb(std::move(completion));
            }
        }

        sort_by_ready(items);
        double batch_finish_ns = batch_start_ns;
        for (auto& item : items) {
            const auto new_ordinal = data_pages + item.ordinal;
            auto completion = hbm.issue(make_request(
                "offload/new-page" + std::to_string(new_ordinal) +
                    "/hbm-write",
                Tier::HBM,
                Op::Write,
                item.ready_ns,
                checked_mul(
                    item.slot,
                    page_size,
                    "capacity-overflow reused HBM address"),
                page_size,
                AddressSpace::Physical,
                config_.trace,
                HeatmapTrafficSource::Workload));
            resident_ordinal.at(item.slot) = new_ordinal;
            batch_finish_ns = std::max(
                batch_finish_ns,
                completion.finish_ns);
            result.offload_offered_latencies_ns.push_back(
                completion.finish_ns - source_arrival(new_ordinal));
            result.offload_service_latencies_ns.push_back(
                completion.finish_ns - batch_start_ns);
            ++result.hbm_user_accesses;
            absorb(std::move(completion));
        }
        prior_batch_finish_ns = batch_finish_ns;
    }
    stats.write_finish_ns = prior_batch_finish_ns;

    // Phase 2: fetch exactly the prefix offloaded in FIFO order. The regular
    // capacity-pressure path streams through the reserved DMA buffer. The
    // layer round-trip path restores each page to its original resident slot,
    // so all pages remain simultaneously resident when the phase completes.
    // Both paths verify availability through an ordinary foreground HBM read.
    const auto restore_original_slots =
        config_.readback_destination ==
        CapacityReadbackDestination::OriginalSlots;
    stats.read_start_ns = stats.write_finish_ns;
    const auto read_batch_limit = config_.transfer_batch_pages;
    double prior_read_batch_finish_ns = stats.read_start_ns;
    for (std::uint64_t batch_begin = 0;
         batch_begin < offload_pages;
         batch_begin += read_batch_limit) {
        const auto batch_count = std::min<std::uint64_t>(
            read_batch_limit,
            offload_pages - batch_begin);
        const auto batch_start_ns = prior_read_batch_finish_ns;
        std::vector<TransferItem> items;
        items.reserve(batch_count);

        for (std::uint64_t offset = 0; offset < batch_count; ++offset) {
            const auto ordinal = batch_begin + offset;
            PhysicalCompletion completion;
            if (hbf) {
                completion = hbf->issue(make_request(
                    "readback/page" + std::to_string(ordinal) +
                        "/hbf-read",
                    Tier::HBF,
                    Op::Read,
                    batch_start_ns,
                    logical_addr(ordinal),
                    page_size,
                    AddressSpace::Logical,
                    config_.trace,
                    HeatmapTrafficSource::DemandFill));
                ++result.hbf_background_accesses;
            } else {
                completion = external->issue(make_request(
                    "readback/page" + std::to_string(ordinal) +
                        "/external-read",
                    Tier::External,
                    Op::Read,
                    batch_start_ns,
                    logical_addr(ordinal),
                    page_size,
                    AddressSpace::Logical,
                    config_.trace,
                    HeatmapTrafficSource::DemandFill));
                ++result.external_background_accesses;
            }
            items.push_back(TransferItem{
                .ordinal = ordinal,
                .slot = restore_original_slots ?
                    ordinal :
                    offset,
                .ready_ns = completion.finish_ns,
            });
            absorb(std::move(completion));
        }

        if (hbf) {
            sort_by_ready(items);
            for (auto& item : items) {
                const auto lpn =
                    logical_addr(item.ordinal) / page_size;
                const auto stack = hbf->stack_for_logical_page(lpn);
                auto completion = links.at(stack).issue(
                    "readback/page" + std::to_string(item.ordinal) +
                        "/d2d-read",
                    Op::Read,
                    item.ready_ns,
                    page_size,
                    config_.trace);
                item.ready_ns = completion.finish_ns;
                absorb(std::move(completion));
            }
        }

        sort_by_ready(items);
        double last_install_issue_ns = batch_start_ns;
        for (auto& item : items) {
            last_install_issue_ns = std::max(
                last_install_issue_ns,
                item.ready_ns);
            const auto install_address =
                restore_original_slots ?
                checked_mul(
                    item.slot,
                    page_size,
                    "capacity-overflow restored HBM slot address") :
                checked_add(
                    data_capacity_bytes,
                    checked_mul(
                        item.slot,
                        page_size,
                        "capacity-overflow read-buffer slot offset"),
                    "capacity-overflow read-buffer address");
            auto completion = hbm.issue(make_request(
                "readback/page" + std::to_string(item.ordinal) +
                    "/hbm-install",
                Tier::HBM,
                Op::Write,
                item.ready_ns,
                install_address,
                page_size,
                AddressSpace::Physical,
                config_.trace,
                HeatmapTrafficSource::DemandFill));
            item.ready_ns = completion.finish_ns;
            ++result.hbm_background_accesses;
            absorb(std::move(completion));
        }

        sort_by_ready(items);
        double batch_finish_ns = batch_start_ns;
        for (auto& item : items) {
            const auto issue_ns = std::max(
                item.ready_ns,
                last_install_issue_ns);
            const auto user_address =
                restore_original_slots ?
                checked_mul(
                    item.slot,
                    page_size,
                    "capacity-overflow restored HBM user address") :
                checked_add(
                    data_capacity_bytes,
                    checked_mul(
                        item.slot,
                        page_size,
                        "capacity-overflow read-buffer user offset"),
                    "capacity-overflow read-buffer user address");
            auto completion = hbm.issue(make_request(
                "readback/page" + std::to_string(item.ordinal) +
                    "/hbm-user-read",
                Tier::HBM,
                Op::Read,
                issue_ns,
                user_address,
                page_size,
                AddressSpace::Physical,
                config_.trace,
                HeatmapTrafficSource::Workload));
            batch_finish_ns = std::max(
                batch_finish_ns,
                completion.finish_ns);
            result.readback_offered_latencies_ns.push_back(
                completion.finish_ns - stats.read_start_ns);
            result.readback_service_latencies_ns.push_back(
                completion.finish_ns - batch_start_ns);
            ++result.hbm_user_accesses;
            absorb(std::move(completion));
        }
        prior_read_batch_finish_ns = batch_finish_ns;
    }
    stats.read_finish_ns = prior_read_batch_finish_ns;
    stats.quiescent_finish_ns = std::max(
        stats.quiescent_finish_ns,
        stats.read_finish_ns);

    if (hbf) {
        auto drain = hbf->drain_pending(
            "capacity-overflow/hbf-drain",
            stats.read_finish_ns,
            config_.trace);
        absorb(std::move(drain));
        result.hbf_stats = hbf->stats();
        result.base_die_link_stats = aggregate_link_stats(links);
    } else {
        result.external_backing_stats = external->stats();
    }
    result.hbm_stats = hbm.stats();
    result.address_heatmap = heatmap.snapshot();

    const auto expected_hbm_write_bytes = checked_add(
        total_write_bytes,
        offload_bytes,
        "capacity-overflow expected HBM writes");
    const auto expected_hbm_read_bytes = checked_mul(
        offload_bytes,
        2,
        "capacity-overflow expected HBM reads");
    if (result.hbm_stats.write_bytes != expected_hbm_write_bytes ||
        result.hbm_stats.read_bytes != expected_hbm_read_bytes ||
        result.hbm_user_accesses != checked_add(
            workload.total_write_pages,
            offload_pages,
            "capacity-overflow expected HBM user accesses") ||
        result.hbm_background_accesses != checked_mul(
            offload_pages,
            2,
            "capacity-overflow expected HBM background accesses")) {
        throw std::runtime_error(
            "capacity-overflow HBM traffic does not conserve");
    }
    if (hbf) {
        if (result.hbf_stats.logical_write_bytes != offload_bytes ||
            result.hbf_stats.logical_read_bytes != offload_bytes ||
            result.hbf_background_accesses != checked_mul(
                offload_pages,
                2,
                "capacity-overflow expected HBF accesses") ||
            result.base_die_link_stats.write_bytes != offload_bytes ||
            result.base_die_link_stats.read_bytes != offload_bytes) {
            throw std::runtime_error(
                "capacity-overflow HBF/link traffic does not conserve");
        }
    } else if (
        result.external_backing_stats.write_bytes != offload_bytes ||
        result.external_backing_stats.read_bytes != offload_bytes ||
        result.external_background_accesses != checked_mul(
            offload_pages,
            2,
            "capacity-overflow expected external accesses")) {
        throw std::runtime_error(
            "capacity-overflow external traffic does not conserve");
    }
    if (stats.fill_finish_ns < stats.first_arrival_ns ||
        stats.offload_start_ns < stats.fill_finish_ns ||
        stats.write_finish_ns < stats.offload_start_ns ||
        stats.read_start_ns != stats.write_finish_ns ||
        stats.read_finish_ns < stats.read_start_ns ||
        stats.quiescent_finish_ns < stats.read_finish_ns ||
        result.fill_write_latencies_ns.size() != data_pages ||
        result.offload_offered_latencies_ns.size() != offload_pages ||
        result.offload_service_latencies_ns.size() != offload_pages ||
        result.readback_offered_latencies_ns.size() != offload_pages ||
        result.readback_service_latencies_ns.size() != offload_pages) {
        throw std::runtime_error(
            "capacity-overflow phase timeline or latency census is inconsistent");
    }
    return result;
}

} // namespace hbfsim::physical::hybrid
