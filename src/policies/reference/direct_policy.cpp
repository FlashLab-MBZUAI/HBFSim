#include "policies/reference/direct_policy.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace hbfsim::policy {
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

PhysicalRequest make_physical(
    std::string id,
    Tier tier,
    const MemoryRequest& op,
    const TraceConfig& trace,
    double arrival_ns,
    std::uint64_t addr,
    AddressSpace address_space,
    HeatmapTrafficSource heatmap_source = HeatmapTrafficSource::Direct) {
    return PhysicalRequest{
        .id = std::move(id),
        .tier = tier,
        .op = op.op,
        .address_space = address_space,
        .trace = trace,
        .arrival_ns = arrival_ns,
        .addr = addr,
        .bytes = op.bytes,
        .stream_id = 0,
        .heatmap_source = heatmap_source,
    };
}

enum class Route {
    Hbm,
    HbfLogical,
    HbfDirectPaged,
};

struct WrittenPlacement {
    std::uint64_t begin_offset = 0;
    std::uint64_t end_offset = 0;
    Route route = Route::Hbm;
    // A cooperative slot generation is unique for one logical-page
    // residency. Multiple foreground writes may contribute to that residency;
    // committing by generation transitions all of them together and cannot
    // accidentally touch a reused physical slot.
    std::optional<std::uint64_t> resident_generation;
    // Physical HBM page base when the current byte range is parked in the
    // cooperative write region. Empty means ordinary address-mapped HBM.
    std::optional<std::uint64_t> hbm_buffer_page_addr;
};

using WrittenPlacementMap =
    std::unordered_map<std::uint64_t, std::vector<WrittenPlacement>>;

Route classify(
    const DirectPolicy& policy,
    const MemoryRequest& op,
    std::uint64_t hbf_page_size,
    bool placement_tracked,
    WrittenPlacementMap& written_placement) {
    if (!policy.build_hbf) {
        return Route::Hbm;
    }
    if (op.op == Op::Read) {
        // Read-after-write coherence: a read of a page this run has WRITTEN
        // must come from where the data actually lives RIGHT NOW (the
        // system's page tables know) — a static physical R partition never
        // received it, and a page still parked in the HBM write region is
        // served from HBM until the HBF write path has accepted its destage.
        // Multi-page reads are split by the caller so every page gets its own
        // placement lookup.
        if (placement_tracked && !written_placement.empty()) {
            const auto page = op.addr / hbf_page_size;
            auto placed = written_placement.find(page);
            if (placed != written_placement.end()) {
                const auto begin = op.addr % hbf_page_size;
                const auto end = checked_add(
                    begin,
                    op.bytes,
                    "written placement read range");
                for (const auto& range : placed->second) {
                    if (range.begin_offset > begin || range.end_offset < end) {
                        continue;
                    }
                    return range.route;
                }
            }
        }
        if (op.addr < policy.read_boundary) {
            return Route::Hbm;
        }
        if (!policy.read_physical_direct) {
            return Route::HbfLogical;
        }
        if (policy.direct_read_kind_filter &&
            op.kind != SemanticKind::ModelWeights &&
            op.kind != SemanticKind::SharedContext &&
            op.kind != SemanticKind::Unknown) {
            return Route::Hbm;
        }
        return Route::HbfDirectPaged;
    }
    switch (policy.write_route) {
    case DirectPolicy::WriteRoute::AllHbm:
        return Route::Hbm;
    case DirectPolicy::WriteRoute::AllHbfLogical:
        return Route::HbfLogical;
    case DirectPolicy::WriteRoute::ReadBoundary:
        return op.addr < policy.read_boundary ? Route::Hbm : Route::HbfLogical;
    }
    throw std::runtime_error("unknown direct-composition write route");
}

std::vector<std::uint64_t> collect_static_initial_ppns(
    const DirectPolicy& policy,
    const host::HbfController& device,
    const std::vector<MemoryRequest>& requests) {
    if (!policy.read_physical_direct) {
        return {};
    }
    const auto page_size = device.config().device.page_size_bytes;
    const auto initial_pages = collect_initial_read_lpns(requests, page_size);
    const std::set<std::uint64_t> initial_image(
        initial_pages.begin(),
        initial_pages.end());
    WrittenPlacementMap no_written_placement;
    std::set<std::uint64_t> ppns;
    for (const auto& request : requests) {
        if (request.op != Op::Read) {
            continue;
        }
        const auto request_last = checked_add(
            request.addr,
            request.bytes - 1,
            "static initial-image read end");
        const auto first_page = request.addr / page_size;
        const auto last_page = request_last / page_size;
        for (auto page = first_page;; ++page) {
            if (initial_image.contains(page)) {
                MemoryRequest page_request = request;
                const auto page_begin = checked_mul(
                    page,
                    page_size,
                    "static initial-image page address");
                page_request.addr = std::max(request.addr, page_begin);
                page_request.bytes = std::min(
                    request_last - page_request.addr + 1,
                    page_size - page_request.addr % page_size);
                if (classify(
                        policy,
                        page_request,
                        page_size,
                        true,
                        no_written_placement) == Route::HbfDirectPaged) {
                    ppns.insert(map_static_hbf_page_addr(device, page) / page_size);
                }
            }
            if (page == last_page) {
                break;
            }
        }
    }
    return {ppns.begin(), ppns.end()};
}

class DirectRequestSource {
public:
    explicit DirectRequestSource(const std::vector<MemoryRequest>& requests)
        : requests_(&requests) {
        validate_memory_requests(requests, "direct-composition");
    }

    explicit DirectRequestSource(SequentialReadWorkload workload)
        : sequential_(workload) {
        if (workload.request_bytes == 0 || workload.request_count == 0) {
            throw std::runtime_error(
                "sequential direct workload request size/count must be positive");
        }
        if (!std::isfinite(workload.first_arrival_ns) ||
            workload.first_arrival_ns < 0.0 ||
            !std::isfinite(workload.interarrival_ns) ||
            workload.interarrival_ns < 0.0) {
            throw std::runtime_error(
                "sequential direct workload arrivals must be finite and non-negative");
        }
        if (workload.request_count >
            static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
            throw std::runtime_error(
                "sequential direct workload request count exceeds uint64_t range");
        }
        const auto count = static_cast<std::uint64_t>(workload.request_count);
        total_bytes_ = checked_mul(
            workload.request_bytes,
            count,
            "sequential direct workload bytes");
        (void)checked_add(
            workload.base_addr,
            total_bytes_ - 1,
            "sequential direct workload end address");
        const double last_arrival = workload.first_arrival_ns +
            static_cast<double>(workload.request_count - 1) *
                workload.interarrival_ns;
        if (!std::isfinite(last_arrival)) {
            throw std::runtime_error(
                "sequential direct workload last arrival is not finite");
        }
        description_.push_back(MemoryRequest{
            .id = "implicit-sequential-read",
            .op = Op::Read,
            .addr = workload.base_addr,
            .bytes = total_bytes_,
            .arrival_ns = workload.first_arrival_ns,
            .index = 0,
        });
        validate_memory_requests(description_, "sequential-direct-description");
    }

    [[nodiscard]] std::size_t size() const {
        return requests_ != nullptr ? requests_->size() : sequential_->request_count;
    }

    [[nodiscard]] MemoryRequest at(std::size_t index) const {
        if (requests_ != nullptr) {
            return requests_->at(index);
        }
        if (index >= sequential_->request_count) {
            throw std::out_of_range("sequential direct workload request index");
        }
        return MemoryRequest{
            .op = Op::Read,
            .addr = sequential_->base_addr +
                static_cast<std::uint64_t>(index) * sequential_->request_bytes,
            .bytes = sequential_->request_bytes,
            .arrival_ns = sequential_->first_arrival_ns +
                static_cast<double>(index) * sequential_->interarrival_ns,
            .index = index,
        };
    }

    [[nodiscard]] bool read_only() const {
        if (sequential_) {
            return true;
        }
        return std::all_of(
            requests_->begin(),
            requests_->end(),
            [](const MemoryRequest& request) { return request.op == Op::Read; });
    }

    [[nodiscard]] bool has_explicit_phases() const {
        return requests_ != nullptr && !requests_->empty() &&
            requests_->front().phase.has_value();
    }

    [[nodiscard]] const std::vector<MemoryRequest>& description_requests() const {
        return requests_ != nullptr ? *requests_ : description_;
    }

    void record_workload_traffic(AddressHeatmap& heatmap) const {
        if (requests_ != nullptr) {
            record_workload_address_traffic(heatmap, *requests_);
            return;
        }
        for (std::size_t i = 0; i < sequential_->request_count; ++i) {
            const auto request = at(i);
            heatmap.record(AddressTrafficRecord{
                .domain = AddressDomain::WorkloadLogical,
                .direction = TrafficDirection::Read,
                .source = HeatmapTrafficSource::Workload,
                .address = request.addr,
                .bytes = request.bytes,
            });
        }
    }

    [[nodiscard]] const std::optional<SequentialReadWorkload>& sequential() const {
        return sequential_;
    }

    [[nodiscard]] std::uint64_t total_bytes() const { return total_bytes_; }

private:
    const std::vector<MemoryRequest>* requests_ = nullptr;
    std::optional<SequentialReadWorkload> sequential_;
    std::vector<MemoryRequest> description_;
    std::uint64_t total_bytes_ = 0;
};

} // namespace

DirectPolicy all_hbm_policy() {
    DirectPolicy policy;
    policy.build_hbf = false;
    policy.write_route = DirectPolicy::WriteRoute::AllHbm;
    policy.hbm_id_prefix = "all-hbm/";
    return policy;
}

DirectPolicy all_hbf_policy() {
    DirectPolicy policy;
    policy.build_hbm = false;
    policy.read_boundary = 0;
    policy.write_route = DirectPolicy::WriteRoute::AllHbfLogical;
    policy.prepopulate = DirectPolicy::Prepopulate::InitialReads;
    policy.drain_hbf = true;
    policy.hbf_id_prefix = "all-hbf/";
    policy.drain_id = "all-hbf/drain";
    return policy;
}

DirectPolicy flat_policy(std::uint64_t flat_hbm_bytes) {
    DirectPolicy policy;
    policy.read_boundary = flat_hbm_bytes;
    policy.write_route = DirectPolicy::WriteRoute::ReadBoundary;
    policy.prepopulate = DirectPolicy::Prepopulate::InitialReadsAtOrAboveReadBoundary;
    policy.drain_hbf = true;
    policy.warn_if_one_sided = true;
    policy.hbm_id_prefix = "flat/hbm/";
    policy.hbf_id_prefix = "flat/hbf/";
    policy.drain_id = "flat/HBF/drain";
    return policy;
}

DirectPolicy read_only_direct_policy(std::uint64_t direct_hbm_bytes) {
    DirectPolicy policy;
    policy.read_boundary = direct_hbm_bytes;
    policy.read_physical_direct = true;
    policy.direct_read_kind_filter = true;
    policy.write_route = DirectPolicy::WriteRoute::AllHbm;
    // Flash is never programmed: no FTL state to prepopulate, nothing to
    // drain.
    policy.hbm_id_prefix = "hbf-static-direct/hbm/";
    policy.direct_id_prefix = "hbf-static-direct/";
    return policy;
}

PolicyRunResult run_direct_policy_impl(
    const DirectPolicy& policy,
    const hbm::HbmConfig& hbm_config,
    const host::HbfConfig& hbf_config,
    const DirectRequestSource& requests,
    const DirectRunKnobs& knobs) {
    const bool tiered_window =
        knobs.max_hbm_outstanding_requests != 0 ||
        knobs.max_hbf_outstanding_requests != 0;
    if (tiered_window && knobs.max_outstanding_requests != 0) {
        throw std::runtime_error(
            "direct-composition shared and tier-specific outstanding "
            "windows are mutually exclusive");
    }
    if (tiered_window && !requests.read_only()) {
        throw std::runtime_error(
            "tier-specific outstanding windows currently require a "
            "read-only direct-composition trace");
    }
    if (knobs.initial_image_requests != nullptr) {
        if (requests.sequential()) {
            throw std::runtime_error(
                "explicit initial-image requests cannot be combined with an "
                "implicit sequential workload");
        }
        validate_memory_requests(
            *knobs.initial_image_requests,
            "direct-composition initial image");
    }
    PolicyRunResult result;
    result.retain_completions = knobs.retain_completions;
    result.service_latencies_ns.reserve(requests.size());
    result.offered_latencies_ns.reserve(requests.size());
    result.source_latencies_ns.reserve(requests.size());
    result.has_hbm = policy.build_hbm || policy.build_hbf;
    result.has_hbf = policy.build_hbf;
    std::optional<AddressHeatmap> address_heatmap;
    // HBF controller storage shares physical HBM even for an all-HBF
    // application policy. Resolve its reservation before assigning regions.
    std::optional<hbm::HbmDevice> hbm;
    std::optional<host::HbfController> hbf;
    if (result.has_hbm) hbm.emplace(hbm_config);
    if (policy.build_hbf) {
        hbf.emplace(hbf_config);
        hbf->attach_hbm_buffer(*hbm);
    }
    if (knobs.address_heatmap_bins != 0) {
        auto heatmap_config = make_composition_address_heatmap_config(
            hbm_config,
            hbf_config,
            requests.description_requests(),
            knobs.address_heatmap_bins);
        if (knobs.hbm_write_buffer_bytes != 0 &&
            hbm && knobs.hbm_write_buffer_bytes <= hbm->application_capacity_bytes()) {
            heatmap_config.domains[
                static_cast<std::size_t>(AddressDomain::HbmPhysical)].regions.push_back(
                    AddressRegion{
                        .name = "cooperative_write_buffer",
                        .kind = AddressRegionKind::CooperativeBuffer,
                        .begin = (hbm->application_capacity_bytes() - knobs.hbm_write_buffer_bytes) /
                            hbf_config.device.page_size_bytes * hbf_config.device.page_size_bytes,
                        .end = (hbm->application_capacity_bytes() - knobs.hbm_write_buffer_bytes) /
                            hbf_config.device.page_size_bytes * hbf_config.device.page_size_bytes +
                            knobs.hbm_write_buffer_bytes,
                    });
        }
        address_heatmap.emplace(std::move(heatmap_config));
        if (hbm) hbm->attach_address_heatmap(*address_heatmap);
        if (hbf) hbf->attach_address_heatmap(*address_heatmap);
        requests.record_workload_traffic(*address_heatmap);
    }
    const auto& population_requests =
        knobs.initial_image_requests != nullptr ?
        *knobs.initial_image_requests :
        requests.description_requests();
    if (hbf && knobs.initial_image_requests != nullptr &&
        policy.prepopulate != DirectPolicy::Prepopulate::None) {
        const auto minimum = policy.prepopulate ==
                DirectPolicy::Prepopulate::InitialReads ?
            std::nullopt :
            std::optional<std::uint64_t>{policy.read_boundary};
        const auto required = collect_initial_read_lpns(
            requests.description_requests(),
            hbf->config().device.page_size_bytes,
            minimum);
        const auto supplied = collect_initial_read_lpns(
            population_requests,
            hbf->config().device.page_size_bytes,
            minimum);
        if (!std::includes(
                supplied.begin(),
                supplied.end(),
                required.begin(),
                required.end())) {
            throw std::runtime_error(
                "explicit initial-image trace does not cover every initial "
                "read page in the executed workload");
        }
    }
    if (hbf && policy.read_physical_direct) {
        if (const auto& sequential = requests.sequential()) {
            const auto page_size = hbf->config().device.page_size_bytes;
            const auto workload_end = checked_add(
                sequential->base_addr,
                requests.total_bytes(),
                "sequential static-direct workload exclusive end");
            const auto first_hbf_addr =
                std::max(sequential->base_addr, policy.read_boundary);
            if (first_hbf_addr < workload_end) {
                const auto first_source_page = first_hbf_addr / page_size;
                const auto last_source_page = (workload_end - 1) / page_size;
                const auto first_block = hbf->decode(
                    map_static_hbf_page_addr(
                        *hbf,
                        first_source_page)).block;
                const auto last_block = hbf->decode(
                    map_static_hbf_page_addr(
                        *hbf,
                        last_source_page)).block;
                if (last_block < first_block) {
                    throw std::runtime_error(
                        "sequential static-direct block extent is not monotonic");
                }
                hbf->reserve_static_physical_block_extent(
                    first_block,
                    last_block - first_block + 1);
            }
        } else {
            hbf->reserve_static_physical_pages(
                collect_static_initial_ppns(
                    policy,
                    *hbf,
                    population_requests));
        }
    }
    if (hbf && policy.prepopulate != DirectPolicy::Prepopulate::None) {
        if (const auto& sequential = requests.sequential()) {
            const auto page_size = hbf->config().device.page_size_bytes;
            const auto workload_end = checked_add(
                sequential->base_addr,
                requests.total_bytes(),
                "sequential direct workload exclusive end");
            const auto first_hbf_addr =
                policy.prepopulate == DirectPolicy::Prepopulate::InitialReads ?
                sequential->base_addr :
                std::max(sequential->base_addr, policy.read_boundary);
            if (first_hbf_addr < workload_end) {
                const auto first_lpn = first_hbf_addr / page_size;
                const auto last_lpn = (workload_end - 1) / page_size;
                hbf->prepopulate_read_only_logical_page_range(
                    first_lpn,
                    last_lpn - first_lpn + 1);
            }
        } else {
            hbf->prepopulate_logical_pages(collect_initial_read_lpns(
                population_requests,
                hbf->config().device.page_size_bytes,
                policy.prepopulate == DirectPolicy::Prepopulate::InitialReads ?
                    std::nullopt :
                    std::optional<std::uint64_t>{policy.read_boundary}));
        }
    }
    // The front-end credit unit is a media page even when this particular
    // policy builds only HBM. Keeping one common transaction granule prevents
    // a large trace record from receiving more implicit parallelism than the
    // equivalent sequence of page-sized records.
    const auto hbf_page_size = hbf_config.device.page_size_bytes;
    if (hbf_page_size == 0) {
        throw std::runtime_error(
            "direct-composition transaction page size must be positive");
    }
    if (policy.build_hbm && policy.build_hbf && hbf_page_size != 0 &&
        policy.read_boundary % hbf_page_size != 0) {
        throw std::runtime_error(
            "direct-composition HBM/HBF boundary must be HBF-page aligned");
    }
    double hbf_admit_frontier_ns = 0.0;
    double hbm_admit_frontier_ns = 0.0;
    const auto reclaim_shared_hbm_history = [&] {
        if (!hbm || !hbf) return;
        const bool application_hbm = policy.build_hbm || knobs.hbm_write_buffer_bytes != 0;
        hbm->advance_buffer_frontier(application_hbm ?
            std::min(hbm_admit_frontier_ns, hbf_admit_frontier_ns) : hbf_admit_frontier_ns);
    };
    bool hbf_has_issued = false;
    // HbfController deliberately rejects time-traveling call order. Composition
    // chains can discover a future link completion before the next foreground
    // request, so make that ordering an explicit admission delay.
    const auto admit_hbf_issue = [&](double nominal_ns) {
        const double admitted_ns = hbf_has_issued ?
            std::max(nominal_ns, hbf_admit_frontier_ns) : nominal_ns;
        hbf_admit_frontier_ns = admitted_ns;
        hbf_has_issued = true;
        reclaim_shared_hbm_history();
        return admitted_ns;
    };
    bool hbm_has_issued = false;
    const auto admit_hbm_issue = [&](double nominal_ns) {
        const double admitted_ns = hbm_has_issued ?
            std::max(nominal_ns, hbm_admit_frontier_ns) : nominal_ns;
        hbm_admit_frontier_ns = admitted_ns;
        hbm_has_issued = true;
        reclaim_shared_hbm_history();
        return admitted_ns;
    };

    // The request stream is admitted in order. A cooperative write that has
    // to wait for a region slot backpressures every following request; those
    // requests cannot make placement decisions at their earlier nominal
    // timestamps after the slot has already been released and reused.
    double foreground_admit_frontier_ns = 0.0;

    // Cooperative write staging state (see DirectRunKnobs).
    const bool coop_write = knobs.hbm_write_buffer_bytes != 0 &&
        policy.build_hbm && policy.build_hbf &&
        policy.write_route != DirectPolicy::WriteRoute::AllHbm;
    result.cooperative_write_controller_present = coop_write;
    if (coop_write && knobs.hbm_write_buffer_bytes > hbm->application_capacity_bytes()) {
        throw std::runtime_error(
            "cooperative HBM write region cannot exceed HBM capacity");
    }
    if (coop_write &&
        (hbf_page_size == 0 ||
         knobs.hbm_write_buffer_bytes < hbf_page_size ||
         knobs.hbm_write_buffer_bytes % hbf_page_size != 0)) {
        throw std::runtime_error(
            "cooperative HBM write region must contain an integral number of "
            "full HBF-page slots");
    }
    const auto application_bytes = hbm ? hbm->application_capacity_bytes() : 0;
    const auto coop_base_addr = coop_write ?
        (application_bytes - knobs.hbm_write_buffer_bytes) / hbf_page_size * hbf_page_size :
        application_bytes;
    if (coop_write && coop_base_addr % hbf_page_size != 0) {
        throw std::runtime_error(
            "cooperative HBM write region must begin on an HBF page boundary");
    }
    if (coop_write && policy.read_boundary > coop_base_addr) {
        throw std::runtime_error(
            "direct-composition HBM boundary overlaps the cooperative write region");
    }
    if (coop_write &&
        (!std::isfinite(knobs.hbm_write_buffer_destage_watermark) ||
         knobs.hbm_write_buffer_destage_watermark < 0.0 ||
         knobs.hbm_write_buffer_destage_watermark > 1.0)) {
        throw std::runtime_error(
            "cooperative HBM write-region destage watermark must be in [0, 1]");
    }
    std::vector<BaseDieLink> destage_links;
    if (coop_write) {
        destage_links.reserve(hbf_config.device.stacks);
        for (std::uint32_t stack = 0; stack < hbf_config.device.stacks; ++stack) {
            destage_links.emplace_back(
                knobs.base_die_link,
                "coop/base_die_link/stack" + std::to_string(stack));
        }
    }
    struct DirtyRange {
        std::uint64_t begin_offset = 0;
        std::uint64_t end_offset = 0;
    };
    struct CoopSlot {
        std::uint64_t logical_page = 0;
        std::uint64_t slot_index = 0;
        std::uint64_t generation = 0;
        std::size_t last_writer_index = 0;
        double buffered_ns = 0.0;
        bool destage_started = false;
        bool destage_scheduled = false;
        std::vector<DirtyRange> dirty_ranges;
    };
    const auto coop_slot_count = coop_write ?
        knobs.hbm_write_buffer_bytes / hbf_page_size : 0;
    std::uint64_t coop_occupancy = 0;
    // Compact free-slot runs avoid allocating metadata proportional to a
    // potentially multi-GiB empty region. Every allocation is exactly one
    // full HBF page; fragmentation therefore cannot make free capacity
    // unusable.
    std::map<std::uint64_t, std::uint64_t> coop_free_slot_runs;
    if (coop_write) {
        coop_free_slot_runs.emplace(0, coop_slot_count);
    }
    // Logical page -> current resident generation, plus generation order for
    // deterministic oldest-resident pressure relief.
    std::unordered_map<std::uint64_t, std::shared_ptr<CoopSlot>>
        coop_resident_pages;
    std::map<std::uint64_t, std::shared_ptr<CoopSlot>>
        coop_slots_by_generation;
    std::uint64_t next_coop_generation = 1;
    // A physical slot is not reusable until every dirty byte in its frozen
    // generation has completed HBM-read -> D2D -> HBF-write admission.
    std::multimap<double, std::shared_ptr<CoopSlot>> coop_releases;
    // Background destage (watermark policy). Resident slots whose destage has
    // neither started nor been scheduled, oldest generation first, plus the
    // chain starts scheduled at the instant a slot's data landed in HBM. A
    // scheduled chain is issued only once the foreground clock reaches it,
    // exactly like the deferred HBF write leg, so background work never
    // time-travels ahead of foreground requests that arrive earlier.
    std::map<std::uint64_t, std::shared_ptr<CoopSlot>> coop_pending_by_generation;
    std::uint64_t coop_pending_slot_bytes = 0;
    std::multimap<double, std::shared_ptr<CoopSlot>> coop_scheduled_destages;
    const double coop_watermark_bytes = coop_write ?
        knobs.hbm_write_buffer_destage_watermark *
            static_cast<double>(knobs.hbm_write_buffer_bytes) :
        0.0;
    // Byte-range placement within each logical page. Sub-page KV writes must
    // not move untouched bytes away from their static HBF source.
    WrittenPlacementMap written_placement;
    const bool placement_tracked =
        policy.build_hbm && policy.build_hbf && hbf_page_size != 0;
    const auto normalize_placement_ranges = [] (
        std::vector<WrittenPlacement>& ranges) {
        std::sort(
            ranges.begin(),
            ranges.end(),
            [](const WrittenPlacement& lhs, const WrittenPlacement& rhs) {
                return lhs.begin_offset < rhs.begin_offset;
            });
        std::vector<WrittenPlacement> normalized;
        normalized.reserve(ranges.size());
        for (auto range : ranges) {
            if (range.begin_offset >= range.end_offset) {
                throw std::runtime_error("empty written-placement range");
            }
            if (!normalized.empty() &&
                normalized.back().end_offset > range.begin_offset) {
                throw std::runtime_error("overlapping written-placement ranges");
            }
            if (!normalized.empty() &&
                normalized.back().end_offset == range.begin_offset &&
                normalized.back().route == range.route &&
                normalized.back().resident_generation ==
                    range.resident_generation &&
                normalized.back().hbm_buffer_page_addr ==
                    range.hbm_buffer_page_addr) {
                normalized.back().end_offset = range.end_offset;
                continue;
            }
            normalized.push_back(std::move(range));
        }
        ranges = std::move(normalized);
    };
    const auto record_write_placement = [&] (
        const MemoryRequest& write_op,
        Route where,
        std::optional<std::uint64_t> hbm_buffer_page_addr = std::nullopt,
        std::optional<std::uint64_t> resident_generation = std::nullopt) {
        if (!placement_tracked) {
            return;
        }
        const auto write_last = checked_add(
            write_op.addr,
            write_op.bytes - 1,
            "written placement end");
        const auto first = write_op.addr / hbf_page_size;
        const auto last = write_last / hbf_page_size;
        if (hbm_buffer_page_addr && first != last) {
            throw std::runtime_error(
                "one cooperative placement update must stay within one HBF page");
        }
        for (auto page = first;; ++page) {
            const auto begin_offset = page == first ?
                write_op.addr % hbf_page_size : 0;
            const auto end_offset = page == last ?
                write_last % hbf_page_size + 1 : hbf_page_size;
            WrittenPlacement replacement{
                .begin_offset = begin_offset,
                .end_offset = end_offset,
                .route = where,
                .resident_generation = resident_generation,
                .hbm_buffer_page_addr = hbm_buffer_page_addr,
            };
            auto& ranges = written_placement[page];
            std::vector<WrittenPlacement> updated;
            updated.reserve(ranges.size() + 2);
            for (const auto& existing : ranges) {
                if (existing.end_offset <= begin_offset ||
                    existing.begin_offset >= end_offset) {
                    updated.push_back(existing);
                    continue;
                }
                if (existing.begin_offset < begin_offset) {
                    auto left = existing;
                    left.end_offset = begin_offset;
                    updated.push_back(std::move(left));
                }
                if (existing.end_offset > end_offset) {
                    auto right = existing;
                    right.begin_offset = end_offset;
                    updated.push_back(std::move(right));
                }
            }
            updated.push_back(std::move(replacement));
            normalize_placement_ranges(updated);
            ranges = std::move(updated);
            if (page == last) {
                break;
            }
        }
    };
    const auto commit_hbf_placement = [&] (
        std::uint64_t page,
        std::uint64_t generation) {
        if (!placement_tracked) {
            return;
        }
        auto found = written_placement.find(page);
        if (found == written_placement.end()) {
            return;
        }
        for (auto& range : found->second) {
            if (range.resident_generation != generation) {
                continue;
            }
            range.route = Route::HbfLogical;
            range.resident_generation.reset();
            range.hbm_buffer_page_addr.reset();
        }
        normalize_placement_ranges(found->second);
    };
    const auto try_allocate_coop_slot = [&]() -> std::optional<std::uint64_t> {
        if (coop_free_slot_runs.empty()) {
            return std::nullopt;
        }
        auto first = coop_free_slot_runs.begin();
        const auto slot_index = first->first;
        const auto remaining = first->second - 1;
        coop_free_slot_runs.erase(first);
        if (remaining != 0) {
            coop_free_slot_runs.emplace(
                checked_add(slot_index, 1, "next cooperative free slot"),
                remaining);
        }
        return slot_index;
    };
    const auto release_coop_slot = [&](std::uint64_t slot_index) {
        if (slot_index >= coop_slot_count) {
            throw std::runtime_error("invalid cooperative HBM slot release");
        }
        std::uint64_t run_begin = slot_index;
        std::uint64_t run_slots = 1;
        auto next = coop_free_slot_runs.lower_bound(slot_index);
        if (next != coop_free_slot_runs.end() && next->first == slot_index) {
            throw std::runtime_error("duplicate cooperative HBM slot release");
        }
        if (next != coop_free_slot_runs.begin()) {
            auto previous = std::prev(next);
            const auto previous_end = checked_add(
                previous->first,
                previous->second,
                "cooperative free-slot run end");
            if (previous_end > slot_index) {
                throw std::runtime_error("overlapping cooperative HBM slot release");
            }
            if (previous_end == slot_index) {
                run_begin = previous->first;
                run_slots = checked_add(
                    run_slots,
                    previous->second,
                    "cooperative merged free-slot run");
                coop_free_slot_runs.erase(previous);
            }
        }
        next = coop_free_slot_runs.lower_bound(slot_index);
        if (next != coop_free_slot_runs.end() &&
            checked_add(run_begin, run_slots, "cooperative released run end") ==
                next->first) {
            run_slots = checked_add(
                run_slots,
                next->second,
                "cooperative merged free-slot run");
            coop_free_slot_runs.erase(next);
        }
        coop_free_slot_runs.emplace(run_begin, run_slots);
    };
    const auto coop_slot_page_addr = [&](const CoopSlot& slot) {
        return checked_add(
            coop_base_addr,
            checked_mul(
                slot.slot_index,
                hbf_page_size,
                "cooperative HBM slot offset"),
            "cooperative HBM slot address");
    };
    const auto merge_dirty_range = [] (
        std::vector<DirtyRange>& ranges,
        std::uint64_t begin_offset,
        std::uint64_t end_offset) {
        if (begin_offset >= end_offset) {
            throw std::runtime_error("empty cooperative dirty range");
        }
        DirtyRange merged{
            .begin_offset = begin_offset,
            .end_offset = end_offset,
        };
        std::vector<DirtyRange> updated;
        updated.reserve(ranges.size() + 1);
        bool inserted = false;
        for (const auto& existing : ranges) {
            if (existing.end_offset < merged.begin_offset) {
                updated.push_back(existing);
                continue;
            }
            if (merged.end_offset < existing.begin_offset) {
                if (!inserted) {
                    updated.push_back(merged);
                    inserted = true;
                }
                updated.push_back(existing);
                continue;
            }
            merged.begin_offset =
                std::min(merged.begin_offset, existing.begin_offset);
            merged.end_offset =
                std::max(merged.end_offset, existing.end_offset);
        }
        if (!inserted) {
            updated.push_back(merged);
        }
        ranges = std::move(updated);
    };
    const auto hbm_addr_for = [&](const MemoryRequest& op) {
        if (placement_tracked && op.op == Op::Read) {
            const auto found = written_placement.find(op.addr / hbf_page_size);
            if (found != written_placement.end()) {
                const auto begin = op.addr % hbf_page_size;
                const auto end = checked_add(
                    begin,
                    op.bytes,
                    "cooperative HBM read range");
                for (const auto& range : found->second) {
                    if (range.route == Route::Hbm &&
                        range.hbm_buffer_page_addr &&
                        range.begin_offset <= begin && range.end_offset >= end) {
                        return checked_add(
                            *range.hbm_buffer_page_addr,
                            begin,
                            "cooperative HBM read address");
                    }
                }
            }
        }
        if (coop_write &&
            (op.addr >= coop_base_addr || op.bytes > coop_base_addr - op.addr)) {
            throw std::runtime_error(
                "ordinary HBM range overlaps the cooperative write region");
        }
        return op.addr;
    };
    struct DestageBatchState {
        std::shared_ptr<CoopSlot> slot;
        std::size_t remaining_segments = 0;
        double release_ns = 0.0;
    };
    struct PendingHbfDestage {
        MemoryRequest request;
        std::uint64_t page = 0;
        std::uint64_t begin_offset = 0;
        double ready_ns = 0.0;
        std::shared_ptr<DestageBatchState> batch;
    };
    std::multimap<double, PendingHbfDestage> pending_hbf_destages;

    // Start only the causally-ready HBM read and D2D stages. The HBF write is
    // a separate ready event; issuing it here would reserve future HBF state
    // ahead of foreground requests that arrive before the link completion.
    const auto start_destage_chain = [&] (
        const std::shared_ptr<CoopSlot>& slot, double start_ns) {
        const auto resident = coop_resident_pages.find(slot->logical_page);
        if (resident == coop_resident_pages.end() ||
            resident->second.get() != slot.get() ||
            slot->destage_started || slot->dirty_ranges.empty()) {
            throw std::runtime_error(
                "invalid cooperative slot destage transition");
        }
        slot->destage_started = true;
        if (!slot->destage_scheduled) {
            if (coop_pending_by_generation.erase(slot->generation) != 1 ||
                coop_pending_slot_bytes < hbf_page_size) {
                throw std::runtime_error(
                    "cooperative pending-destage index diverged");
            }
            coop_pending_slot_bytes -= hbf_page_size;
        }
        std::uint64_t destaged_bytes = 0;
        const double effective_start_ns = std::max(start_ns, slot->buffered_ns);
        auto batch = std::make_shared<DestageBatchState>();
        batch->slot = slot;
        batch->release_ns = effective_start_ns;
        const auto logical_page_addr = checked_mul(
            slot->logical_page,
            hbf_page_size,
            "cooperative destage logical page address");
        const auto slot_page_addr = coop_slot_page_addr(*slot);
        for (const auto& dirty : slot->dirty_ranges) {
            if (dirty.begin_offset >= dirty.end_offset ||
                dirty.end_offset > hbf_page_size) {
                throw std::runtime_error(
                    "cooperative dirty range exceeds its resident page");
            }
            const auto segment_bytes = dirty.end_offset - dirty.begin_offset;
            const auto logical_addr = checked_add(
                logical_page_addr,
                dirty.begin_offset,
                "cooperative destage logical byte address");
            MemoryRequest segment{
                .id = "coop-destage",
                .op = Op::Write,
                .addr = logical_addr,
                .bytes = segment_bytes,
                .arrival_ns = effective_start_ns,
                .index = slot->last_writer_index,
                .kind = SemanticKind::Unknown,
                .label = {},
            };
            MemoryRequest region_read_op = segment;
            region_read_op.op = Op::Read;
            const auto hbm_issue_ns = admit_hbm_issue(effective_start_ns);
            auto region_read = hbm->issue(make_physical(
                policy.hbf_id_prefix + "op" +
                    std::to_string(slot->last_writer_index) +
                    "/destage-read/page" +
                    std::to_string(slot->logical_page) + "/off" +
                    std::to_string(dirty.begin_offset),
                Tier::HBM,
                region_read_op,
                knobs.trace,
                hbm_issue_ns,
                checked_add(
                    slot_page_addr,
                    dirty.begin_offset,
                    "cooperative HBM dirty-range address"),
                AddressSpace::Physical,
                HeatmapTrafficSource::Destage));
            const double read_done = region_read.finish_ns;
            batch->release_ns = std::max(batch->release_ns, read_done);
            result.hbm_background_accesses = checked_add(
                result.hbm_background_accesses,
                1,
                "cooperative HBM background access count");
            add_background_completion(result, std::move(region_read));
            const auto link_stack =
                hbf->stack_for_logical_page(slot->logical_page);
            auto link_out = destage_links.at(link_stack).issue(
                policy.hbf_id_prefix + "op" +
                    std::to_string(slot->last_writer_index) +
                    "/destage-link/page" +
                    std::to_string(slot->logical_page) + "/off" +
                    std::to_string(dirty.begin_offset),
                Op::Write,
                read_done,
                segment_bytes,
                knobs.trace);
            const double link_done = link_out.finish_ns;
            add_background_completion(result, std::move(link_out));
            pending_hbf_destages.emplace(link_done, PendingHbfDestage{
                .request = std::move(segment),
                .page = slot->logical_page,
                .begin_offset = dirty.begin_offset,
                .ready_ns = link_done,
                .batch = batch,
            });
            batch->remaining_segments++;
            result.hbm_write_buffer_destaged_bytes = checked_add(
                result.hbm_write_buffer_destaged_bytes,
                segment_bytes,
                "cooperative dirty-union destaged byte count");
            destaged_bytes = checked_add(
                destaged_bytes,
                segment_bytes,
                "cooperative slot destaged byte count");
        }
        if (batch->remaining_segments == 0) {
            throw std::runtime_error("cooperative destage produced no HBF segments");
        }
        return destaged_bytes;
    };

    const auto process_one_hbf_destage = [&]() {
        if (pending_hbf_destages.empty()) {
            throw std::runtime_error("no pending cooperative HBF destage event");
        }
        auto found = pending_hbf_destages.begin();
        PendingHbfDestage ready = std::move(found->second);
        pending_hbf_destages.erase(found);
        const auto hbf_issue_ns = admit_hbf_issue(ready.ready_ns);
        auto destage = hbf->issue(make_physical(
            policy.hbf_id_prefix + "op" +
                std::to_string(ready.batch->slot->last_writer_index) +
                "/destage/page" + std::to_string(ready.page) + "/off" +
                std::to_string(ready.begin_offset),
            Tier::HBF,
            ready.request,
            knobs.trace,
            hbf_issue_ns,
            ready.request.addr,
            AddressSpace::Logical,
            HeatmapTrafficSource::Destage));
        ready.batch->release_ns = std::max(
            ready.batch->release_ns,
            destage.finish_ns);
        result.hbf_background_accesses = checked_add(
            result.hbf_background_accesses,
            1,
            "cooperative HBF background access count");
        add_background_completion(result, std::move(destage));
        if (ready.batch->remaining_segments == 0) {
            throw std::runtime_error("cooperative HBF destage batch underflow");
        }
        ready.batch->remaining_segments--;
        if (ready.batch->remaining_segments == 0) {
            coop_releases.emplace(
                ready.batch->release_ns,
                ready.batch->slot);
        }
    };

    const auto process_hbf_destages_through = [&](double now_ns, bool all = false) {
        while (!pending_hbf_destages.empty() &&
               (all || pending_hbf_destages.begin()->first <= now_ns)) {
            process_one_hbf_destage();
        }
    };

    const auto process_coop_releases_through = [&] (
        double now_ns,
        bool all = false) {
        while (!coop_releases.empty() &&
               (all || coop_releases.begin()->first <= now_ns)) {
            auto found = coop_releases.begin();
            auto slot = std::move(found->second);
            coop_releases.erase(found);
            const auto resident =
                coop_resident_pages.find(slot->logical_page);
            const auto by_generation =
                coop_slots_by_generation.find(slot->generation);
            if (resident == coop_resident_pages.end() ||
                resident->second.get() != slot.get() ||
                by_generation == coop_slots_by_generation.end() ||
                by_generation->second.get() != slot.get() ||
                !slot->destage_started || coop_occupancy < hbf_page_size) {
                throw std::runtime_error(
                    "invalid cooperative slot release transition");
            }
            commit_hbf_placement(slot->logical_page, slot->generation);
            coop_resident_pages.erase(resident);
            coop_slots_by_generation.erase(by_generation);
            coop_occupancy -= hbf_page_size;
            release_coop_slot(slot->slot_index);
        }
    };

    // Scheduled background chains are issued when the foreground clock
    // reaches their start instant, mirroring the deferred HBF write leg.
    const auto process_scheduled_destages_through = [&] (
        double now_ns,
        bool all = false) {
        while (!coop_scheduled_destages.empty() &&
               (all || coop_scheduled_destages.begin()->first <= now_ns)) {
            auto found = coop_scheduled_destages.begin();
            const double start_ns = found->first;
            auto slot = std::move(found->second);
            coop_scheduled_destages.erase(found);
            // Slot pressure may have started (and even released) this
            // generation first; the schedule entry is then stale.
            if (!slot->destage_started) {
                (void)start_destage_chain(slot, start_ns);
            }
        }
    };

    // Watermark policy: once the resident slots whose destage has neither
    // started nor been scheduled exceed the configured fraction of the
    // region, schedule the oldest of them (generation order) until the
    // pending footprint fits again. Each chain starts when its slot's last
    // foreground write has landed in HBM.
    const auto schedule_destages_to_watermark = [&]() {
        while (!coop_pending_by_generation.empty() &&
               static_cast<double>(coop_pending_slot_bytes) >
                   coop_watermark_bytes) {
            auto oldest = coop_pending_by_generation.begin();
            auto slot = oldest->second;
            coop_pending_by_generation.erase(oldest);
            if (slot->destage_scheduled || slot->destage_started ||
                coop_pending_slot_bytes < hbf_page_size) {
                throw std::runtime_error(
                    "cooperative watermark scheduler saw an inconsistent slot");
            }
            coop_pending_slot_bytes -= hbf_page_size;
            slot->destage_scheduled = true;
            coop_scheduled_destages.emplace(slot->buffered_ns, slot);
        }
    };

    const auto wait_for_slot_release = [&] (
        std::shared_ptr<CoopSlot> target,
        double& now_ns) {
        const auto still_resident = [&]() {
            const auto found =
                coop_resident_pages.find(target->logical_page);
            return found != coop_resident_pages.end() &&
                found->second.get() == target.get();
        };
        if (!still_resident()) {
            return;
        }
        if (!target->destage_started) {
            start_destage_chain(
                target,
                std::max(now_ns, target->buffered_ns));
        }
        while (still_resident()) {
            process_scheduled_destages_through(now_ns);
            process_coop_releases_through(now_ns);
            if (!still_resident()) {
                break;
            }
            const double next_scheduled_ns = coop_scheduled_destages.empty() ?
                std::numeric_limits<double>::infinity() :
                coop_scheduled_destages.begin()->first;
            const double next_hbf_ready_ns = pending_hbf_destages.empty() ?
                std::numeric_limits<double>::infinity() :
                pending_hbf_destages.begin()->first;
            const double next_release_ns = coop_releases.empty() ?
                std::numeric_limits<double>::infinity() :
                coop_releases.begin()->first;
            if (!coop_scheduled_destages.empty() &&
                next_scheduled_ns <= next_hbf_ready_ns &&
                next_scheduled_ns <= next_release_ns) {
                now_ns = std::max(now_ns, next_scheduled_ns);
                process_scheduled_destages_through(now_ns);
                continue;
            }
            if (!pending_hbf_destages.empty() &&
                next_hbf_ready_ns <= next_release_ns) {
                now_ns = std::max(now_ns, next_hbf_ready_ns);
                process_one_hbf_destage();
                continue;
            }
            if (std::isfinite(next_release_ns)) {
                now_ns = std::max(now_ns, next_release_ns);
                process_coop_releases_through(now_ns);
                continue;
            }
            throw std::runtime_error(
                "cooperative slot is in flight without a completion event");
        }
    };

    const auto allocate_resident_slot = [&] (
        std::uint64_t logical_page) -> std::shared_ptr<CoopSlot> {
        const auto slot_index = try_allocate_coop_slot();
        if (!slot_index) {
            throw std::runtime_error(
                "cooperative slot allocation requested without free capacity");
        }
        const auto generation = next_coop_generation;
        next_coop_generation = checked_add(
            next_coop_generation,
            1,
            "cooperative slot generation");
        auto slot = std::make_shared<CoopSlot>();
        slot->logical_page = logical_page;
        slot->slot_index = *slot_index;
        slot->generation = generation;
        if (!coop_resident_pages.emplace(logical_page, slot).second ||
            !coop_slots_by_generation.emplace(generation, slot).second ||
            !coop_pending_by_generation.emplace(generation, slot).second) {
            throw std::runtime_error("duplicate cooperative slot residency");
        }
        coop_occupancy = checked_add(
            coop_occupancy,
            hbf_page_size,
            "cooperative full-page slot occupancy");
        coop_pending_slot_bytes = checked_add(
            coop_pending_slot_bytes,
            hbf_page_size,
            "cooperative pending-destage occupancy");
        if (coop_occupancy > knobs.hbm_write_buffer_bytes) {
            throw std::runtime_error(
                "cooperative HBM write-region capacity invariant violated");
        }
        result.hbm_write_buffer_peak_bytes =
            std::max(result.hbm_write_buffer_peak_bytes, coop_occupancy);
        return slot;
    };

    const auto ensure_writable_slot = [&] (
        std::uint64_t logical_page,
        double& now_ns) -> std::shared_ptr<CoopSlot> {
        // Release events at or before the foreground instant are authoritative
        // before consulting the page index; otherwise a reused physical slot
        // could be reached through stale resident metadata.
        process_scheduled_destages_through(now_ns);
        process_coop_releases_through(now_ns);
        auto resident = coop_resident_pages.find(logical_page);
        if (resident != coop_resident_pages.end()) {
            auto slot = resident->second;
            if (!slot->destage_started) {
                // A scheduled-but-not-started generation still merges: its
                // dirty snapshot is taken only when the chain starts.
                return slot;
            }
            // Never mutate a generation after its dirty snapshot left HBM.
            // Conservatively wait for the whole chain and allocate a fresh
            // generation for this later write.
            wait_for_slot_release(slot, now_ns);
        }
        while (coop_free_slot_runs.empty()) {
            if (coop_slots_by_generation.empty()) {
                throw std::runtime_error(
                    "cooperative region has no free or resident slot");
            }
            wait_for_slot_release(
                coop_slots_by_generation.begin()->second,
                now_ns);
        }
        return allocate_resident_slot(logical_page);
    };

    struct CooperativeWriteOutcome {
        double admit_ns = 0.0;
        double finish_ns = 0.0;
    };
    const auto issue_cooperative_write = [&] (
        const MemoryRequest& write_op,
        std::size_t index) -> CooperativeWriteOutcome {
        double admit_ns = write_op.arrival_ns;
        double finish_ns = admit_ns;
        auto segment_begin = write_op.addr;
        auto remaining = write_op.bytes;
        while (remaining != 0) {
            const auto page = segment_begin / hbf_page_size;
            const auto begin_offset = segment_begin % hbf_page_size;
            const auto segment_bytes = std::min(
                remaining,
                hbf_page_size - begin_offset);
            const auto end_offset = checked_add(
                begin_offset,
                segment_bytes,
                "cooperative dirty-range end");
            auto slot = ensure_writable_slot(page, admit_ns);
            const auto slot_page_addr = coop_slot_page_addr(*slot);
            MemoryRequest segment = write_op;
            segment.addr = segment_begin;
            segment.bytes = segment_bytes;
            segment.arrival_ns = admit_ns;
            record_write_placement(
                segment,
                Route::Hbm,
                slot_page_addr,
                slot->generation);
            merge_dirty_range(
                slot->dirty_ranges,
                begin_offset,
                end_offset);
            slot->last_writer_index = index;
            const auto hbm_issue_ns = admit_hbm_issue(admit_ns);
            auto foreground = hbm->issue(make_physical(
                policy.hbf_id_prefix + "op" + std::to_string(index) +
                    "/hbm-write-region/page" + std::to_string(page) +
                    "/off" + std::to_string(begin_offset),
                Tier::HBM,
                segment,
                knobs.trace,
                hbm_issue_ns,
                checked_add(
                    slot_page_addr,
                    begin_offset,
                    "cooperative HBM foreground write address"),
                AddressSpace::Physical,
                HeatmapTrafficSource::CooperativeBuffer));
            finish_ns = std::max(finish_ns, foreground.finish_ns);
            slot->buffered_ns =
                std::max(slot->buffered_ns, foreground.finish_ns);
            add_user_completion(result, std::move(foreground), true, false);
            remaining -= segment_bytes;
            if (remaining != 0) {
                segment_begin = checked_add(
                    segment_begin,
                    segment_bytes,
                    "next cooperative foreground write segment");
            }
        }
        result.hbm_write_buffer_user_write_bytes = checked_add(
            result.hbm_write_buffer_user_write_bytes,
            write_op.bytes,
            "cooperative user write byte count");
        foreground_admit_frontier_ns =
            std::max(foreground_admit_frontier_ns, admit_ns);
        // The wait for a free slot is this write's own front-end back-pressure
        // (the caller records its admission after it); the counters keep it
        // visible separately from the shared admission-wait sums.
        if (admit_ns > write_op.arrival_ns) {
            result.hbm_write_buffer_full_waits = checked_add(
                result.hbm_write_buffer_full_waits,
                1,
                "cooperative write-region wait count");
            const double updated_wait_ns = result.hbm_write_buffer_wait_ns +
                (admit_ns - write_op.arrival_ns);
            if (!std::isfinite(updated_wait_ns)) {
                throw std::runtime_error(
                    "cooperative write-region wait time is not finite");
            }
            result.hbm_write_buffer_wait_ns = updated_wait_ns;
        }
        schedule_destages_to_watermark();
        return CooperativeWriteOutcome{
            .admit_ns = admit_ns,
            .finish_ns = finish_ns,
        };
    };

    ClosedLoopWindow shared_window{knobs.max_outstanding_requests};
    ClosedLoopWindow hbm_window{knobs.max_hbm_outstanding_requests};
    ClosedLoopWindow hbf_window{knobs.max_hbf_outstanding_requests};
    std::optional<std::uint64_t> active_phase;
    double completed_phase_frontier_ns = 0.0;
    double active_phase_finish_ns = 0.0;
    const auto admit_transaction = [&](Route route, double nominal_ns) {
        if (!tiered_window) {
            return shared_window.admit(nominal_ns);
        }
        return route == Route::Hbm ?
            hbm_window.admit(nominal_ns) :
            hbf_window.admit(nominal_ns);
    };
    const auto complete_transaction = [&](Route route, double finish_ns) {
        if (active_phase) {
            active_phase_finish_ns =
                std::max(active_phase_finish_ns, finish_ns);
        }
        if (tiered_window) {
            if (route == Route::Hbm) {
                hbm_window.complete(finish_ns);
            } else {
                hbf_window.complete(finish_ns);
            }
        } else {
            shared_window.complete(finish_ns);
        }
    };
    struct ParentCursor {
        std::size_t index = 0;
        MemoryRequest op;
        double source_arrival_ns = 0.0;
        double offered_arrival_ns = 0.0;
        double dependency_ready_ns = 0.0;
        std::optional<double> first_transaction_admit_ns = std::nullopt;
        double finish_ns = 0.0;
        std::set<std::uint64_t> direct_pages_issued{};
        std::uint64_t segment_begin = 0;
        std::uint64_t remaining = 0;
    };
    // This work sum can contain O(100M) terms whose magnitudes grow with the
    // closed-loop backlog. Accumulating directly into the public double loses
    // enough low bits to make an otherwise exact per-request latency identity
    // fail at report time. Keep the reduction in extended precision and round
    // once after the scheduler has quiesced.
    long double front_end_admission_wait_work_ns = 0.0L;

    const auto record_first_transaction_admission =
        [&](ParentCursor& parent, double admitted_ns) {
            if (parent.first_transaction_admit_ns) {
                return;
            }
            parent.first_transaction_admit_ns = admitted_ns;
            const double admission_wait_ns =
                admitted_ns - parent.offered_arrival_ns;
            if (admission_wait_ns < 0.0 ||
                !std::isfinite(admission_wait_ns)) {
                throw std::runtime_error(
                    "direct front-end admission wait must be finite and "
                    "non-negative");
            }
            if (admission_wait_ns == 0.0) {
                return;
            }
            result.front_end_admission_waited_ops = checked_add(
                result.front_end_admission_waited_ops,
                1,
                "direct front-end admission waited operation count");
            front_end_admission_wait_work_ns +=
                static_cast<long double>(admission_wait_ns);
            if (!std::isfinite(front_end_admission_wait_work_ns) ||
                front_end_admission_wait_work_ns >
                    static_cast<long double>(
                        std::numeric_limits<double>::max())) {
                throw std::runtime_error(
                    "direct front-end admission wait work is not finite");
            }
            result.front_end_admission_max_wait_ns = std::max(
                result.front_end_admission_max_wait_ns,
                admission_wait_ns);
        };

    const auto issue_next_transaction = [&](ParentCursor& parent) {
        if (parent.remaining == 0) {
            throw std::runtime_error(
                "direct scheduler issued a completed parent");
        }
        const auto page = parent.segment_begin / hbf_page_size;
        const auto offset = parent.segment_begin % hbf_page_size;
        auto segment_bytes = std::min(
            parent.remaining,
            hbf_page_size - offset);

        // Runtime placement is byte-range precise. Split at the next overlay
        // boundary so each issued transaction has one owner.
        if (placement_tracked && parent.op.op == Op::Read) {
            const auto found = written_placement.find(page);
            if (found != written_placement.end()) {
                for (const auto& range : found->second) {
                    if (range.end_offset <= offset) {
                        continue;
                    }
                    if (range.begin_offset <= offset) {
                        segment_bytes = std::min(
                            segment_bytes,
                            range.end_offset - offset);
                    } else {
                        segment_bytes = std::min(
                            segment_bytes,
                            range.begin_offset - offset);
                    }
                    break;
                }
            }
        }

        MemoryRequest segment = parent.op;
        segment.addr = parent.segment_begin;
        segment.bytes = segment_bytes;
        auto route = classify(
            policy,
            segment,
            hbf_page_size,
            placement_tracked,
            written_placement);

        // A direct HBF page fetch covers every untouched byte range in that
        // page. Sparse HBM/HBF overlays may split the logical parent, but they
        // must not trigger duplicate full-page media reads.
        const bool duplicate_direct_page =
            route == Route::HbfDirectPaged &&
            !parent.direct_pages_issued.insert(page).second;
        if (!duplicate_direct_page) {
            const double window_admit_ns =
                admit_transaction(route, parent.dependency_ready_ns);
            // A full cooperative region back-pressures every later request.
            // That delay is front-end admission wait (recorded per route
            // below, after this max), not service time of the delayed
            // request.
            segment.arrival_ns = std::max(
                window_admit_ns,
                foreground_admit_frontier_ns);

            if (coop_write) {
                process_scheduled_destages_through(segment.arrival_ns);
                process_hbf_destages_through(segment.arrival_ns);
                process_coop_releases_through(segment.arrival_ns);
                const auto current_route = classify(
                    policy,
                    segment,
                    hbf_page_size,
                    placement_tracked,
                    written_placement);
                if (tiered_window && current_route != route) {
                    throw std::runtime_error(
                        "tier route changed after transaction-window "
                        "admission");
                }
                route = current_route;
            }

            double transaction_finish_ns = segment.arrival_ns;
            switch (route) {
            case Route::Hbm: {
                record_first_transaction_admission(parent, segment.arrival_ns);
                if (segment.op == Op::Write) {
                    record_write_placement(segment, Route::Hbm);
                }
                const auto hbm_issue_ns =
                    admit_hbm_issue(segment.arrival_ns);
                // One HBM issue discipline for every composition: each page
                // transaction is issued and completed synchronously, so the
                // completion that returns its credit is known immediately
                // and all-hbm, flat, and direct-read exercise the same
                // controller path.
                auto completion = hbm->issue(make_physical(
                    policy.hbm_id_prefix + "op" +
                        std::to_string(parent.index) + "/page" +
                        std::to_string(page) + "/off" +
                        std::to_string(offset),
                    Tier::HBM,
                    segment,
                    knobs.trace,
                    hbm_issue_ns,
                    hbm_addr_for(segment),
                    AddressSpace::Logical));
                transaction_finish_ns = completion.finish_ns;
                add_user_completion(
                    result,
                    std::move(completion),
                    true,
                    false);
                break;
            }
            case Route::HbfLogical: {
                if (segment.op == Op::Write && !coop_write) {
                    record_write_placement(
                        segment,
                        Route::HbfLogical);
                }
                if (coop_write && segment.op == Op::Write) {
                    // Waiting for a staging slot is admission wait for this
                    // write too; its service starts once a slot is held.
                    const auto staged = issue_cooperative_write(
                        segment,
                        parent.index);
                    record_first_transaction_admission(
                        parent, staged.admit_ns);
                    transaction_finish_ns = staged.finish_ns;
                } else {
                    record_first_transaction_admission(
                        parent, segment.arrival_ns);
                    const auto hbf_issue_ns =
                        admit_hbf_issue(segment.arrival_ns);
                    auto completion = hbf->issue(make_physical(
                        policy.hbf_id_prefix + "op" +
                            std::to_string(parent.index) + "/page" +
                            std::to_string(page) + "/off" +
                            std::to_string(offset),
                        Tier::HBF,
                        segment,
                        knobs.trace,
                        hbf_issue_ns,
                        segment.addr,
                        AddressSpace::Logical));
                    transaction_finish_ns = completion.finish_ns;
                    add_user_completion(
                        result,
                        std::move(completion),
                        false,
                        true);
                }
                break;
            }
            case Route::HbfDirectPaged: {
                record_first_transaction_admission(parent, segment.arrival_ns);
                const auto hbf_issue_ns =
                    admit_hbf_issue(segment.arrival_ns);
                auto completion = hbf->issue(make_physical(
                    policy.direct_id_prefix + "op" +
                        std::to_string(parent.index) + "/page" +
                        std::to_string(page),
                    Tier::HBF,
                    segment,
                    knobs.trace,
                    hbf_issue_ns,
                    static_hbf_addr_for_byte(
                        *hbf,
                        parent.segment_begin),
                    AddressSpace::Physical));
                transaction_finish_ns = completion.finish_ns;
                add_user_completion(
                    result,
                    std::move(completion),
                    false,
                    true);
                result.hbf_direct_user_ops = checked_add(
                    result.hbf_direct_user_ops,
                    1,
                    "direct static HBF operation count");
                result.hbf_static_read_bytes = checked_add(
                    result.hbf_static_read_bytes,
                    hbf_page_size,
                    "direct static HBF read bytes");
                break;
            }
            }

            parent.finish_ns = std::max(
                parent.finish_ns,
                transaction_finish_ns);
            complete_transaction(route, transaction_finish_ns);
        }

        parent.remaining -= segment_bytes;
        if (parent.remaining != 0) {
            parent.segment_begin = checked_add(
                parent.segment_begin,
                segment_bytes,
                "next direct-composition page transaction");
        }
    };

    const auto finish_parent = [&](ParentCursor& parent) {
        if (parent.remaining != 0 ||
            !parent.first_transaction_admit_ns) {
            throw std::runtime_error(
                "direct parent completed without issuing every physical "
                "transaction");
        }
        record_user_op(
            result,
            parent.op,
            parent.finish_ns,
            *parent.first_transaction_admit_ns,
            parent.offered_arrival_ns,
            parent.source_arrival_ns);
    };

    // Run one explicit phase at a time. Within a phase, every ready parent
    // receives one page transaction per round. This preserves the physical
    // transaction credit contract while preventing a multi-hundred-MiB DMA
    // record from monopolizing the shared window ahead of thousands of
    // independent requests at the same timestamp.
    const bool stream_one_page_sequential =
        requests.sequential() &&
        requests.sequential()->request_bytes == hbf_page_size &&
        requests.sequential()->base_addr % hbf_page_size == 0;
    if (stream_one_page_sequential) {
        // Every parent owns exactly one media-page transaction, so the
        // generic phase-wide round-robin queue is observationally identical
        // to a streaming loop. Avoid retaining tens of millions of parent
        // cursors for capacity-scale sequential reads.
        for (std::size_t index = 0; index < requests.size(); ++index) {
            auto op = requests.at(index);
            const auto source_arrival_ns = op.arrival_ns;
            result.first_offered_arrival_ns = std::min(
                result.first_offered_arrival_ns,
                op.arrival_ns);
            result.last_offered_arrival_ns = std::max(
                result.last_offered_arrival_ns,
                op.arrival_ns);
            ParentCursor parent{
                .index = index,
                .op = std::move(op),
                .source_arrival_ns = source_arrival_ns,
                .offered_arrival_ns = source_arrival_ns,
                .dependency_ready_ns = source_arrival_ns,
                .finish_ns = source_arrival_ns,
                .segment_begin = requests.at(index).addr,
                .remaining = hbf_page_size,
            };
            issue_next_transaction(parent);
            finish_parent(parent);
        }
    } else {
      std::size_t phase_begin = 0;
      bool first_phase_group = true;
      while (phase_begin < requests.size()) {
        const auto first_request = requests.at(phase_begin);
        const auto group_phase = first_request.phase;
        std::size_t phase_end = phase_begin + 1;
        if (requests.has_explicit_phases()) {
            while (phase_end < requests.size() &&
                   requests.at(phase_end).phase == group_phase) {
                ++phase_end;
            }
            if (!group_phase) {
                throw std::runtime_error(
                    "direct trace mixes phased and unphased requests");
            }
            if (first_phase_group) {
                active_phase = group_phase;
            } else {
                completed_phase_frontier_ns =
                    active_phase_finish_ns;
                active_phase = group_phase;
                active_phase_finish_ns =
                    completed_phase_frontier_ns;
                result.phase_barriers = checked_add(
                    result.phase_barriers,
                    1,
                    "direct phase-barrier count");
            }
        } else {
            phase_end = requests.size();
        }

        std::vector<ParentCursor> parents;
        parents.reserve(phase_end - phase_begin);
        for (auto index = phase_begin; index < phase_end; ++index) {
            auto op = requests.at(index);
            const double source_arrival_ns = op.arrival_ns;
            if (group_phase) {
                op.arrival_ns = std::max(
                    op.arrival_ns,
                    completed_phase_frontier_ns);
                const double dependency_wait_ns =
                    op.arrival_ns - source_arrival_ns;
                if (dependency_wait_ns > 0.0) {
                    result.phase_dependency_waited_ops = checked_add(
                        result.phase_dependency_waited_ops,
                        1,
                        "direct phase-dependency waited operation count");
                    const double updated_wait_ns =
                        result.phase_dependency_wait_work_ns +
                        dependency_wait_ns;
                    if (!std::isfinite(updated_wait_ns)) {
                        throw std::runtime_error(
                            "direct phase-dependency wait work is not finite");
                    }
                    result.phase_dependency_wait_work_ns =
                        updated_wait_ns;
                    result.phase_dependency_max_wait_ns = std::max(
                        result.phase_dependency_max_wait_ns,
                        dependency_wait_ns);
                }
            }
            result.first_offered_arrival_ns = std::min(
                result.first_offered_arrival_ns,
                op.arrival_ns);
            result.last_offered_arrival_ns = std::max(
                result.last_offered_arrival_ns,
                op.arrival_ns);
            const auto offered_arrival_ns = op.arrival_ns;
            const auto segment_begin = op.addr;
            const auto remaining = op.bytes;
            parents.push_back(ParentCursor{
                .index = index,
                .op = std::move(op),
                .source_arrival_ns = source_arrival_ns,
                .offered_arrival_ns = offered_arrival_ns,
                .dependency_ready_ns = offered_arrival_ns,
                .finish_ns = offered_arrival_ns,
                .segment_begin = segment_begin,
                .remaining = remaining,
            });
        }

        // Round-robin is safe for independent reads, but it must not let a
        // later request overtake an earlier overlapping write (or let a write
        // overtake an earlier overlapping read). Build page-granular parent
        // dependencies only for phases that contain writes. A dependent
        // parent starts after every relevant predecessor completes; unrelated
        // pages retain full round-robin concurrency.
        std::vector<std::size_t> unresolved_dependencies(parents.size(), 0);
        std::vector<std::vector<std::size_t>> dependents(parents.size());
        if (std::any_of(
                parents.begin(),
                parents.end(),
                [](const ParentCursor& parent) {
                    return parent.op.op == Op::Write;
                })) {
            struct PageOrderingState {
                std::optional<std::size_t> last_writer;
                std::vector<std::size_t> readers_since_write;
            };
            std::unordered_map<std::uint64_t, PageOrderingState> page_order;
            for (std::size_t i = 0; i < parents.size(); ++i) {
                const auto& op = parents[i].op;
                const auto last_addr = checked_add(
                    op.addr,
                    op.bytes - 1,
                    "direct dependency range end");
                const auto first_page = op.addr / hbf_page_size;
                const auto last_page = last_addr / hbf_page_size;
                std::set<std::size_t> predecessors;
                for (auto page = first_page;; ++page) {
                    auto& state = page_order[page];
                    if (op.op == Op::Read) {
                        if (state.last_writer) {
                            predecessors.insert(*state.last_writer);
                        }
                        state.readers_since_write.push_back(i);
                    } else {
                        if (state.last_writer) {
                            predecessors.insert(*state.last_writer);
                        }
                        predecessors.insert(
                            state.readers_since_write.begin(),
                            state.readers_since_write.end());
                        state.readers_since_write.clear();
                        state.last_writer = i;
                    }
                    if (page == last_page) {
                        break;
                    }
                }
                unresolved_dependencies[i] = predecessors.size();
                for (const auto predecessor : predecessors) {
                    dependents[predecessor].push_back(i);
                }
            }
        }

        std::deque<std::size_t> ready;
        std::size_t next_parent = 0;
        std::vector<bool> arrived(parents.size(), false);
        std::vector<bool> completed(parents.size(), false);
        const auto admission_frontier = [&]() {
            if (!tiered_window) {
                return shared_window.last_admit_ns;
            }
            return std::max(
                hbm_window.last_admit_ns,
                hbf_window.last_admit_ns);
        };
        const auto activate_ready = [&](double frontier_ns) {
            while (next_parent < parents.size() &&
                   parents[next_parent].offered_arrival_ns <=
                       frontier_ns) {
                arrived[next_parent] = true;
                if (unresolved_dependencies[next_parent] == 0) {
                    ready.push_back(next_parent);
                }
                ++next_parent;
            }
        };

        std::size_t completed_parents = 0;
        while (completed_parents < parents.size()) {
            if (ready.empty()) {
                if (next_parent >= parents.size()) {
                    throw std::runtime_error(
                        "direct parent dependency scheduler deadlocked");
                }
                const auto next_arrival =
                    parents[next_parent].offered_arrival_ns;
                activate_ready(next_arrival);
                if (ready.empty()) {
                    throw std::runtime_error(
                        "direct parent dependency became ready before its "
                        "predecessor");
                }
            } else {
                activate_ready(admission_frontier());
            }
            const auto parent_index = ready.front();
            ready.pop_front();
            auto& parent = parents[parent_index];

            issue_next_transaction(parent);

            if (parent.remaining == 0) {
                finish_parent(parent);
                completed[parent_index] = true;
                ++completed_parents;
                for (const auto dependent : dependents[parent_index]) {
                    if (completed[dependent] ||
                        unresolved_dependencies[dependent] == 0) {
                        throw std::runtime_error(
                            "invalid direct parent dependency release");
                    }
                    --unresolved_dependencies[dependent];
                    parents[dependent].dependency_ready_ns = std::max(
                        parents[dependent].dependency_ready_ns,
                        parent.finish_ns);
                    if (unresolved_dependencies[dependent] == 0 &&
                        arrived[dependent]) {
                        ready.push_back(dependent);
                    }
                }
            } else {
                ready.push_back(parent_index);
            }
            activate_ready(admission_frontier());
        }

          phase_begin = phase_end;
          first_phase_group = false;
      }
    }

    if (coop_write) {
        // Everything destaged from here on is drain work: it starts after the
        // last user completion. Chains the watermark scheduled but the
        // foreground clock never reached start at their own instants (the
        // HBM issue frontier orders them after the last user request); every
        // other parked slot resumes at the user frontier. Snapshot the
        // generation map because starting a chain intentionally leaves the
        // slot resident until its HBF acceptance event completes.
        const double resume_ns = result.user_finish_ns;
        while (!coop_scheduled_destages.empty()) {
            auto found = coop_scheduled_destages.begin();
            const double start_ns = found->first;
            auto slot = std::move(found->second);
            coop_scheduled_destages.erase(found);
            if (!slot->destage_started) {
                result.hbm_write_buffer_drain_destaged_bytes = checked_add(
                    result.hbm_write_buffer_drain_destaged_bytes,
                    start_destage_chain(slot, start_ns),
                    "cooperative drain-destaged byte count");
            }
        }
        std::vector<std::shared_ptr<CoopSlot>> remaining_slots;
        remaining_slots.reserve(coop_slots_by_generation.size());
        for (const auto& [generation, slot] : coop_slots_by_generation) {
            (void)generation;
            remaining_slots.push_back(slot);
        }
        for (const auto& slot : remaining_slots) {
            if (!slot->destage_started) {
                result.hbm_write_buffer_drain_destaged_bytes = checked_add(
                    result.hbm_write_buffer_drain_destaged_bytes,
                    start_destage_chain(
                        slot,
                        std::max(slot->buffered_ns, resume_ns)),
                    "cooperative drain-destaged byte count");
            }
        }
        process_hbf_destages_through(
            std::numeric_limits<double>::infinity(), true);
        process_coop_releases_through(
            std::numeric_limits<double>::infinity(), true);
        if (!coop_resident_pages.empty() ||
            !coop_slots_by_generation.empty() ||
            !coop_pending_by_generation.empty() ||
            !coop_scheduled_destages.empty() ||
            coop_pending_slot_bytes != 0 ||
            !pending_hbf_destages.empty() || !coop_releases.empty() ||
            coop_occupancy != 0 || coop_free_slot_runs.size() != 1 ||
            coop_free_slot_runs.begin()->first != 0 ||
            coop_free_slot_runs.begin()->second != coop_slot_count) {
            throw std::runtime_error(
                "cooperative write region did not fully release at drain");
        }
    }

    if (policy.warn_if_one_sided &&
        (result.hbm_user_accesses + result.hbm_background_accesses == 0 ||
         result.hbf_user_accesses + result.hbf_background_accesses == 0)) {
        result.warnings.push_back(
            "flat policy did not exercise both HBM and HBF; adjust --flat-hbm-bytes");
    }
    if (policy.drain_hbf && hbf) {
        auto drain = hbf->drain_pending(
            policy.drain_id,
            admit_hbf_issue(result.finish_ns),
            knobs.trace);
        if (drain.physical_bytes != 0 || !drain.spans.empty()) {
            add_background_completion(result, std::move(drain));
        }
    }
    if (hbm) {
        result.hbm_stats = hbm->stats();
    }
    if (hbf) {
        result.hbf_stats = hbf->stats();
        result.hbf_wear_snapshot = hbf->wear_snapshot_json();
    }
    if (result.hbf_static_read_bytes != checked_mul(
            result.hbf_direct_user_ops,
            hbf_page_size,
            "static-direct HBF byte accounting")) {
        throw std::runtime_error(
            "static-direct HBF bytes do not match direct page transactions");
    }
    if (const auto& sequential = requests.sequential();
        sequential &&
        sequential->request_bytes == hbf_page_size &&
        sequential->base_addr % hbf_page_size == 0) {
        // Capacity sweeps rely on a stronger contract than the generic
        // per-request accounting: an aligned page stream must visit the
        // complete low-address HBM prefix before its first HBF page. Check
        // the split from the workload geometry itself so a stale config
        // boundary or an off-by-one classifier cannot silently produce a
        // plausible-looking aggregate.
        const auto request_count =
            static_cast<std::uint64_t>(sequential->request_count);
        std::uint64_t expected_hbm_requests = 0;
        if (!policy.build_hbf) {
            expected_hbm_requests = request_count;
        } else if (policy.build_hbm &&
                   sequential->base_addr < policy.read_boundary) {
            const auto prefix_bytes =
                policy.read_boundary - sequential->base_addr;
            expected_hbm_requests = std::min(
                request_count,
                prefix_bytes / hbf_page_size);
        }
        const auto expected_hbf_requests =
            request_count - expected_hbm_requests;
        if (result.hbm_user_accesses != expected_hbm_requests ||
            result.hbf_user_accesses != expected_hbf_requests) {
            throw std::runtime_error(
                "aligned sequential read did not conserve its exact "
                "HBM-prefix/HBF-suffix transaction split");
        }
        const auto expected_hbm_bytes = checked_mul(
            expected_hbm_requests,
            hbf_page_size,
            "aligned sequential HBM byte split");
        const auto expected_hbf_bytes = checked_mul(
            expected_hbf_requests,
            hbf_page_size,
            "aligned sequential HBF byte split");
        if ((result.has_hbm &&
             result.hbm_stats.read_bytes != expected_hbm_bytes +
                 result.hbm_stats.controller_buffer_read_bytes) ||
            (!result.has_hbm && expected_hbm_bytes != 0) ||
            (result.has_hbf &&
             checked_add(
                 result.hbf_stats.logical_read_bytes,
                 result.hbf_static_read_bytes,
                 "aligned sequential HBF logical/direct byte split") !=
                 expected_hbf_bytes) ||
            (!result.has_hbf && expected_hbf_bytes != 0)) {
            throw std::runtime_error(
                "aligned sequential read did not conserve its exact "
                "HBM-prefix/HBF-suffix byte split");
        }
    }
    for (const auto& link : destage_links) {
        const auto& link_stats = link.stats();
        result.base_die_link_stats.read_transfers = checked_add(
            result.base_die_link_stats.read_transfers,
            link_stats.read_transfers,
            "direct base-link read transfer count");
        result.base_die_link_stats.write_transfers = checked_add(
            result.base_die_link_stats.write_transfers,
            link_stats.write_transfers,
            "direct base-link write transfer count");
        result.base_die_link_stats.read_bytes = checked_add(
            result.base_die_link_stats.read_bytes,
            link_stats.read_bytes,
            "direct base-link read bytes");
        result.base_die_link_stats.write_bytes = checked_add(
            result.base_die_link_stats.write_bytes,
            link_stats.write_bytes,
            "direct base-link write bytes");
        result.base_die_link_stats.read_queue_wait_ns += link_stats.read_queue_wait_ns;
        result.base_die_link_stats.write_queue_wait_ns += link_stats.write_queue_wait_ns;
        result.base_die_link_stats.read_busy_ns += link_stats.read_busy_ns;
        result.base_die_link_stats.write_busy_ns += link_stats.write_busy_ns;
        result.base_die_link_stats.read_fixed_latency_work_ns +=
            link_stats.read_fixed_latency_work_ns;
        result.base_die_link_stats.write_fixed_latency_work_ns +=
            link_stats.write_fixed_latency_work_ns;
        result.base_die_link_stats.first_arrival_ns = std::min(
            result.base_die_link_stats.first_arrival_ns,
            link_stats.first_arrival_ns);
        result.base_die_link_stats.finish_ns = std::max(
            result.base_die_link_stats.finish_ns,
            link_stats.finish_ns);
    }
    result.base_die_link_stats.links = destage_links.size();

    result.front_end_admission_wait_work_ns =
        static_cast<double>(front_end_admission_wait_work_ns);
    if ((result.front_end_admission_waited_ops == 0) !=
            (result.front_end_admission_wait_work_ns == 0.0) ||
        (result.front_end_admission_waited_ops == 0) !=
            (result.front_end_admission_max_wait_ns == 0.0) ||
        result.front_end_admission_max_wait_ns >
            result.front_end_admission_wait_work_ns) {
        throw std::runtime_error(
            "direct front-end admission accounting did not conserve wait work");
    }
    if (!std::isfinite(result.first_offered_arrival_ns) ||
        !std::isfinite(result.last_offered_arrival_ns) ||
        result.last_offered_arrival_ns < result.first_offered_arrival_ns) {
        throw std::runtime_error(
            "direct offered-arrival frontiers are invalid");
    }
    if ((result.phase_dependency_waited_ops == 0) !=
            (result.phase_dependency_wait_work_ns == 0.0) ||
        (result.phase_dependency_waited_ops == 0) !=
            (result.phase_dependency_max_wait_ns == 0.0) ||
        result.phase_dependency_max_wait_ns >
            result.phase_dependency_wait_work_ns) {
        throw std::runtime_error(
            "direct phase-dependency wait accounting did not conserve work");
    }
    if (address_heatmap) {
        result.address_heatmap = address_heatmap->snapshot();
    }

    return result;
}

PolicyRunResult run_direct_policy(
    const DirectPolicy& policy,
    const hbm::HbmConfig& hbm_config,
    const host::HbfConfig& hbf_config,
    const std::vector<MemoryRequest>& requests,
    const DirectRunKnobs& knobs) {
    return run_direct_policy_impl(
        policy,
        hbm_config,
        hbf_config,
        DirectRequestSource{requests},
        knobs);
}

PolicyRunResult run_direct_policy(
    const DirectPolicy& policy,
    const hbm::HbmConfig& hbm_config,
    const host::HbfConfig& hbf_config,
    const SequentialReadWorkload& workload,
    const DirectRunKnobs& knobs) {
    return run_direct_policy_impl(
        policy,
        hbm_config,
        hbf_config,
        DirectRequestSource{workload},
        knobs);
}

} // namespace hbfsim::policy
