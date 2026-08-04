#pragma once

#include "physical/hbf/hbf_device.hpp"
#include "physical/hbm/hbm_device.hpp"
#include "physical/physical_types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <vector>

namespace hbfsim::physical::hybrid {

// One backing-tier identity shared by every composition. Placement and
// lifecycle policies may differ, but selecting HBF versus an external
// backing must not create parallel enum contracts.
enum class BackingTier {
    Hbf,
    External,
};

[[nodiscard]] const char* to_string(BackingTier tier);

// Optional semantic hint on a request. Plain address-only traces are the
// first-class citizen: every composition must be fully defined without kinds;
// a policy may CONSUME a kind as a placement hint but never require one.
enum class SemanticKind {
    Unknown,
    ModelWeights,
    SharedContext,
    GeneratedContext,
    Scratch,
    Metadata,
};

[[nodiscard]] std::string to_string(SemanticKind kind);

// A memory operation as the composition layer sees it.
struct MemoryRequest {
    std::string id;
    Op op = Op::Read;
    std::uint64_t addr = 0;
    std::uint64_t bytes = 0;
    double arrival_ns = 0.0;
    std::size_t index = 0;
    SemanticKind kind = SemanticKind::Unknown;
    std::string label;
    // Optional execution phase for dependency-aware direct compositions.
    // Requests in one phase may overlap; phase N+1 is not offered to a
    // device until every request in phase N has completed. This is distinct
    // from `layer`: phase is a scheduling dependency, while layer identifies
    // a streaming-buffer residency window.
    std::optional<std::uint64_t> phase;
    // Optional execution-layer identity for the ping-pong layer streamer.
    // Layer ids must be nondecreasing. If every request omits this field, the
    // complete trace is one explicit streaming window so plain R/W traces stay
    // first-class inputs.
    std::optional<std::uint64_t> layer;
    // Optional GPU execution interval attached to one or more requests in a
    // streaming layer. Repeated declarations in the same layer must agree. The
    // interval overlaps next-layer prefetch and extends the layer dependency
    // frontier; omitting it keeps the run explicitly memory-only.
    std::optional<double> compute_ns;
};

// Fail closed before any device state is mutated. Composition traces are an
// ordered event stream: requests must have positive, representable ranges and
// finite, nondecreasing nominal arrivals.
void validate_memory_requests(
    const std::vector<MemoryRequest>& requests,
    std::string_view context);

// One bounded heatmap configuration shared by every composition. Logical
// domains span the full configured memory plus the offered trace extent;
// physical domains span their actual media capacities. Semantic regions are
// derived only from request labels and never affect routing.
[[nodiscard]] AddressHeatmapConfig make_composition_address_heatmap_config(
    const hbm::HbmConfig& hbm_config,
    const hbf::HbfConfig& hbf_config,
    const std::vector<MemoryRequest>& requests,
    std::size_t bin_count,
    std::uint64_t external_capacity_bytes = 0);
void record_workload_address_traffic(
    AddressHeatmap& heatmap,
    const std::vector<MemoryRequest>& requests);

// Completion-order bounded physical-transaction window. One page-sized
// foreground transaction consumes one credit until it completes; when all
// credits are in use, the next transaction waits only for the earliest
// completion, independent of parent request boundaries or issue order. 0 is
// unbounded. Device/controller resources may add local backpressure.
struct ClosedLoopWindow {
    explicit ClosedLoopWindow(std::size_t window_limit) : limit(window_limit) {}
    std::size_t limit = 0;
    std::priority_queue<
        double,
        std::vector<double>,
        std::greater<double>> inflight;
    double last_admit_ns = 0.0;
    double admit(double nominal_ns) {
        double admitted_ns = std::max(nominal_ns, last_admit_ns);
        if (limit != 0) {
            // Retire every request already complete at this causal frontier.
            // Removing all ties is important: simultaneous completions return
            // all of their credits before requests admitted at the same time.
            while (!inflight.empty() && inflight.top() <= admitted_ns) {
                inflight.pop();
            }
            while (inflight.size() >= limit) {
                admitted_ns = inflight.top();
                do {
                    inflight.pop();
                } while (!inflight.empty() && inflight.top() <= admitted_ns);
            }
        }
        last_admit_ns = admitted_ns;
        return admitted_ns;
    }
    void complete(double finish_ns) {
        if (limit != 0) {
            inflight.push(finish_ns);
        }
    }
};

// Total physical page slots addressable by an HBF geometry. Throws when the
// geometry cannot be represented by uint64_t.
[[nodiscard]] std::uint64_t hbf_page_capacity(const hbf::HbfDevice& device);

// Static placement of a source page onto the HBF fabric for direct physical
// reads (no FTL). This is a capacity-checked bijection: every source page in
// [0, hbf_page_capacity) maps to exactly one physical page. It deliberately
// reuses the managed FTL's page-striped stack owner and per-stack round-robin
// data-plane order so static-direct access removes metadata work without
// changing data parallelism.
[[nodiscard]] std::uint64_t map_static_hbf_page_addr(
    const hbf::HbfDevice& device,
    std::uint64_t source_page);

// Inverse of map_static_hbf_page_addr. The input must be an in-capacity,
// page-aligned physical byte address produced by the static mapping domain.
[[nodiscard]] std::uint64_t unmap_static_hbf_page_addr(
    const hbf::HbfDevice& device,
    std::uint64_t physical_page_addr);

// Byte-granular variant: page placement via map_static_hbf_page_addr plus
// the in-page offset.
[[nodiscard]] std::uint64_t static_hbf_addr_for_byte(
    const hbf::HbfDevice& device,
    std::uint64_t source_byte_addr);

// Page-granular initial logical image inferred without looking into the
// future: a page is present initially iff a read observes at least one byte
// not covered by earlier writes. Thus a full-page write-then-read page begins
// absent, while a partial write followed by an untouched-byte read preserves
// the old page. min_addr restricts which qualifying reads are materialized in
// the selected tier, while written-range tracking remains global.
[[nodiscard]] std::vector<std::uint64_t> collect_initial_read_lpns(
    const std::vector<MemoryRequest>& requests,
    std::uint64_t page_size,
    std::optional<std::uint64_t> min_addr = std::nullopt);

// PER-STACK link: every HBF stack talks to the HBM side over its own
// exploratory base-die D2D interface; a single shared link would be a
// different topology.
struct BaseDieLinkConfig {
    double read_bandwidth_GBps = 512.0;
    double write_bandwidth_GBps = 128.0;
    double latency_ns = 20.0;
};

struct BaseDieLinkStats {
    std::uint64_t read_transfers = 0;
    std::uint64_t write_transfers = 0;
    std::uint64_t read_bytes = 0;
    std::uint64_t write_bytes = 0;
    double read_queue_wait_ns = 0.0;
    double write_queue_wait_ns = 0.0;
    double read_busy_ns = 0.0;
    double write_busy_ns = 0.0;
    // Propagation/response latency is not serialization occupancy, but it is
    // still additive per-transfer stage work and must remain visible.
    double read_fixed_latency_work_ns = 0.0;
    double write_fixed_latency_work_ns = 0.0;
    double first_arrival_ns = std::numeric_limits<double>::infinity();
    double finish_ns = 0.0;
    // Number of physical links aggregated into this record; utilization is
    // busy time over links x wall clock (fraction of aggregate capacity).
    std::uint64_t links = 1;

    [[nodiscard]] double read_utilization() const;
    [[nodiscard]] double write_utilization() const;
    [[nodiscard]] double active_span_ns() const;
};

class BaseDieLink {
public:
    explicit BaseDieLink(BaseDieLinkConfig config, std::string entity = "hybrid/base_die_link");

    [[nodiscard]] PhysicalCompletion issue(
        std::string id,
        Op op,
        double arrival_ns,
        std::uint64_t bytes,
        TraceConfig trace);

    [[nodiscard]] const BaseDieLinkStats& stats() const { return stats_; }

private:
    BaseDieLinkConfig config_;
    std::string entity_;
    BaseDieLinkStats stats_;
    double read_ready_ns_ = 0.0;
    double write_ready_ns_ = 0.0;
};

// Unified per-composition measurement. user_finish_ns is the completion of
// the last USER op (the throughput divisor); finish_ns additionally covers
// background work and the end-of-run drain, and the difference is the drain
// tail.
struct CompositionRunResult {
    // Three non-overlapping latency views for every parent request:
    // service starts when its first physical page transaction receives a
    // credit; offered additionally includes front-end credit wait; source
    // additionally includes explicit phase/dependency wait. User-facing
    // online latency is offered-to-completion.
    std::vector<double> service_latencies_ns;
    std::vector<double> offered_latencies_ns;
    std::vector<double> source_latencies_ns;
    std::vector<PhysicalCompletion> completions;
    std::vector<std::string> warnings;

    // Full completion objects are diagnostic payload, not simulation state.
    // Large summary-only runs can disable their retention after each
    // completion has contributed to accounting and device statistics.
    bool retain_completions = true;

    std::uint64_t ops = 0;
    std::uint64_t reads = 0;
    std::uint64_t writes = 0;
    std::uint64_t logical_bytes = 0;
    std::uint64_t hbm_user_accesses = 0;
    std::uint64_t hbf_user_accesses = 0;
    std::uint64_t hbf_direct_user_ops = 0;
    std::uint64_t hbm_background_accesses = 0;
    std::uint64_t hbf_background_accesses = 0;
    // Bytes read from the HBF fabric via static physical placement (full
    // pages, even when the requested segment is partial).
    std::uint64_t hbf_static_read_bytes = 0;
    double finish_ns = 0.0;
    double user_finish_ns = 0.0;
    // Device-facing offered-arrival frontiers after explicit phase
    // dependencies are applied, but before the closed-loop window admits the
    // first page transaction of each parent request.
    double first_offered_arrival_ns =
        std::numeric_limits<double>::infinity();
    double last_offered_arrival_ns = 0.0;
    std::uint64_t phase_barriers = 0;
    std::uint64_t phase_dependency_waited_ops = 0;
    double phase_dependency_wait_work_ns = 0.0;
    double phase_dependency_max_wait_ns = 0.0;
    // Delay from a parent's offered arrival to admission of its first physical
    // page transaction. Later child-transaction waits remain inside the
    // parent's service latency.
    std::uint64_t front_end_admission_waited_ops = 0;
    double front_end_admission_wait_work_ns = 0.0;
    double front_end_admission_max_wait_ns = 0.0;
    // True only when this composition actually instantiates the finite HBM
    // cooperative-write controller. A configured-but-inapplicable knob must
    // not make consumers infer a controller that was never on the path.
    bool cooperative_write_controller_present = false;
    // Cooperative-write staging counters (zero unless the HBM write region
    // is enabled): host/user bytes accepted into cooperative slots (overlaps
    // included), coalesced dirty-union bytes actually destaged HBM->HBF, the
    // region's peak full-page occupancy, and write ops that waited for space.
    std::uint64_t hbm_write_buffer_user_write_bytes = 0;
    std::uint64_t hbm_write_buffer_destaged_bytes = 0;
    std::uint64_t hbm_write_buffer_peak_bytes = 0;
    std::uint64_t hbm_write_buffer_full_waits = 0;
    double hbm_write_buffer_wait_ns = 0.0;
    // Aggregate base-die D2D traffic over the composition's per-stack
    // links (direct compositions: the coop destage chain's write leg).
    // Zero when no link is exercised.
    BaseDieLinkStats base_die_link_stats;
    bool has_hbm = false;
    bool has_hbf = false;
    hbm::HbmStats hbm_stats;
    hbf::HbfStats hbf_stats;
    std::optional<AddressHeatmapSnapshot> address_heatmap;
};

// Record one USER op's completion (ops/reads/writes/bytes, all three latency
// views, and both finish frontiers).
void record_user_op(
    CompositionRunResult& result,
    const MemoryRequest& request,
    double finish_ns,
    double service_start_ns,
    double offered_arrival_ns,
    double source_arrival_ns);

// Absorb a device completion. User-visible completions count toward the
// medium's user access tally; background ones (drain, background flushes)
// only move finish_ns.
void add_user_completion(
    CompositionRunResult& result,
    PhysicalCompletion completion,
    bool hbm,
    bool hbf);
void add_background_completion(
    CompositionRunResult& result,
    PhysicalCompletion completion);

} // namespace hbfsim::physical::hybrid
