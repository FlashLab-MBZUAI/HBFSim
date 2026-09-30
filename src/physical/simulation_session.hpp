#pragma once

#include "physical/address_heatmap.hpp"
#include "physical/external/external_backing_device.hpp"
#include "physical/base_die_link.hpp"
#include "host/hbf_controller.hpp"
#include "physical/hbm/hbm_device.hpp"
#include "physical/physical_types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hbfsim::physical {

// Raised for a batch the session rejected before touching any device state
// (malformed transaction, unknown dependency, cycle, duplicate id, ...). The
// session stays usable; the protocol layer answers with an error record
// instead of terminating the process.
class SimulationInputError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// This is the execution boundary for workload-driven studies. Every routing
// and migration decision has already been made by an external remapper. The
// executor deliberately has no model, layer, phase, KV, or semantic-kind
// fields to inspect.
enum class SimulationTarget {
    Hbm,
    HbfLogical,
    HbfStatic,
    HbfPhysical,
    D2dHbfToHbm,
    D2dHbmToHbf,
    External,
    HostDram,
    // Direct HBF<->external lane: the per-stack package-exit path that
    // bypasses HBM entirely. Like the D2D targets this is a pure transport
    // stage; the HBF and external device work of a migration is expressed
    // by explicit HBF_*/EXTERNAL transactions chained with dependencies.
    DirectHbfToExternal,
    DirectExternalToHbf,
    Barrier,
};

[[nodiscard]] const char* to_string(SimulationTarget target);

struct SimulationTransaction {
    std::string id;
    SimulationTarget target = SimulationTarget::Hbm;
    Op op = Op::Read;
    std::uint64_t addr = 0;
    std::uint64_t bytes = 0;
    double issue_ns = 0.0;
    // Only Barrier transactions may have non-zero duration. A barrier is a
    // generic dependency/timer record, not a workload semantic annotation.
    double duration_ns = 0.0;
    std::vector<std::string> dependencies;
    // D2D and DIRECT links are per HBF stack. Other targets must use
    // stack 0.
    std::uint32_t stack = 0;
};

inline constexpr std::size_t kSimulationTargetCount = 11;

struct SimulationTargetCensus {
    std::uint64_t transactions = 0;
    std::uint64_t bytes = 0;
};

// HBM requests the session handed to the controller in a batch and the
// DRAM bursts they decomposed into; device-level receipts live in HbmStats.
struct SimulationHbmEngineStats {
    std::uint64_t requests = 0;
    std::uint64_t bursts = 0;
};

// Transaction-level timing observed at the simulation-session boundary.
// Work sums may include overlapping transactions; active_span_ns (derived
// from first_arrival_ns and finish_ns) is the wall-clock denominator for an
// effective traffic rate.
struct SimulationCompletionStats {
    std::uint64_t transactions = 0;
    std::uint64_t logical_bytes = 0;
    std::uint64_t physical_bytes = 0;
    double queue_wait_work_ns = 0.0;
    double service_work_ns = 0.0;
    double latency_work_ns = 0.0;
    double min_latency_ns = std::numeric_limits<double>::infinity();
    double max_latency_ns = 0.0;
    double first_arrival_ns = std::numeric_limits<double>::infinity();
    double finish_ns = 0.0;
    // Internal Kahan corrections for queue, service and latency respectively.
    // Keep these with the ledger when copying/resetting it; they are not wire
    // metrics. Trailing public storage preserves aggregate initialization.
    std::array<double, 3> work_compensation_ns{};

    void record_completion(const PhysicalCompletion& completion);
};

inline constexpr std::size_t kSimulationOperationCount = 2;

struct SimulationDeviceSnapshot {
    bool has_hbm = false;
    bool has_hbf = false;
    bool has_external = false;
    bool has_host_dram = false;
    hbm::HbmStats hbm;
    host::HbfStats hbf;
    external::ExternalBackingStats external;
    external::ExternalBackingStats host_dram;
    BaseDieLinkStats base_die_link;
    BaseDieLinkStats hbf_external_direct_link;
};

struct SimulationDrainResult {
    bool has_hbf = false;
    double serving_frontier_ns = 0.0;
    double finish_ns = 0.0;
    PhysicalCompletion hbf_completion;
    SimulationDeviceSnapshot device_before;
    SimulationDeviceSnapshot device_after;
    host::HbfQuiescenceStats quiescence;
};

struct SimulationObservationResult {
    SimulationDeviceSnapshot devices;
    host::HbfQuiescenceStats quiescence;
};

struct SimulationCheckpointResult {
    std::string checkpoint_id;
    std::size_t sequence = 0;
    SimulationDrainResult persistence;
};

struct SimulationInvalidationResult {
    host::HbfLogicalInvalidationResult invalidation;
    SimulationDeviceSnapshot device_before;
    SimulationDeviceSnapshot device_after;
};

struct SimulationCrashResult {
    std::string crash_id;
    std::size_t completed_batches = 0;
    std::size_t completed_checkpoints = 0;
    double completed_frontier_ns = 0.0;
    SimulationDeviceSnapshot device_at_injection;
    host::HbfQuiescenceStats quiescence;
};

struct SimulationBatchResult {
    struct ObservedTiming {
        std::string id;
        double start_ns = 0.0;
        double finish_ns = 0.0;
    };
    std::string batch_id;
    std::size_t sequence = 0;
    std::uint64_t transactions = 0;
    std::uint64_t memory_transactions = 0;
    std::uint64_t barriers = 0;
    std::uint64_t dependency_edges = 0;
    // Sum of bytes on every emitted low-level transaction. Migration stages
    // intentionally count again; canonical logical-byte conservation is
    // proven by the remapper receipt before this execution boundary.
    std::uint64_t transaction_bytes = 0;
    // Input issue times are offsets within a batch. This is the persistent
    // device-time origin at which those offsets were applied.
    double batch_origin_ns = 0.0;
    double first_issue_ns = 0.0;
    // Completion of every transaction in the batch, detached work included.
    double finish_ns = 0.0;
    // Completion of the caller's frontier (see SimulationBatchOptions). The
    // next batch is rebased onto this value, never onto finish_ns.
    double blocking_finish_ns = 0.0;
    // Number of transactions whose completion defined blocking_finish_ns.
    std::uint64_t frontier_transactions = 0;
    std::array<SimulationTargetCensus, kSimulationTargetCount> by_target{};
    std::array<
        std::array<SimulationCompletionStats, kSimulationOperationCount>,
        kSimulationTargetCount> completion_by_target{};
    SimulationHbmEngineStats hbm_engine;
    // Per-batch delta, not the persistent device lifetime total.
    host::HbfReadEngineStats hbf_read_engine;
    SimulationDeviceSnapshot device_before;
    SimulationDeviceSnapshot device_after;
    // Empty when the batch was run with record_completions == false.
    std::vector<PhysicalCompletion> completions;
    // Only explicitly requested DAG nodes, including compute barriers.
    std::vector<ObservedTiming> observed_timings;
};

// Per-batch execution options supplied by the caller.
struct SimulationBatchOptions {
    // Transactions whose completion gates the next batch. Absent (nullopt)
    // means every transaction of the batch; an empty list means none, so the
    // next origin advances only past the batch's last arrival. Every named
    // id must belong to the batch. Work outside the frontier (for example a
    // background offload write) still executes, still records into the
    // dependency table, and still contributes to finish_ns.
    std::optional<std::vector<std::string>> frontier;
    // Complete set of earlier transaction ids that later batches may still
    // name as dependencies. The session forgets every id older than
    // kDependencyWindowBatches completed batches unless it is retained, and
    // a retained id must be resolvable when it is listed (this batch, the
    // window, or the current retained set). A list replaces the retained
    // set (an empty list clears it); absent (nullopt) leaves the set
    // unchanged, so a producer that declares what it holds is not undone
    // by another producer's batches on the same session. The table stays
    // bounded by what callers declared.
    std::optional<std::vector<std::string>> retain;
    // Keep per-transaction physical completions in the result. Aggregate
    // timing and device accounting are recorded either way.
    bool record_completions = true;
    // Lightweight native timing observations; never changes scheduling or
    // frontier/retention semantics. IDs must belong to this batch.
    std::vector<std::string> observe;
};

// Completed batches whose transaction ids stay resolvable as dependencies
// without an explicit retain declaration.
inline constexpr std::size_t kDependencyWindowBatches = 2;

struct SimulationSessionConfig {
    bool enable_hbm = true;
    bool enable_hbf = true;
    bool enable_external = false;
    hbm::HbmConfig hbm;
    host::HbfConfig hbf;
    external::ExternalBackingConfig external;
    // Separate CPU memory attachment, sharing the session clock and HBM
    // dependencies with the primary external device.
    std::optional<external::ExternalBackingConfig> host_dram;
    BaseDieLinkConfig base_die_link;
    // Optional per-stack direct lane between the HBF base die and the
    // external device, bypassing HBM. The lane's read direction carries
    // HBF-to-external offload traffic and its write direction carries
    // external-to-HBF restore traffic. It has no default envelope: a caller
    // that wants the lane declares its bandwidth and latency explicitly, and
    // the session instantiates it only when both tiers are enabled.
    std::optional<BaseDieLinkConfig> hbf_external_direct_link;
    // Immutable physical data occupy blocks [0, N) on every HBF plane.
    // This is a low-level placement declaration supplied by the remapper.
    std::uint32_t static_hbf_blocks_per_plane = 0;
    // Directly programmed publication data occupy the immediately following
    // blocks [static, static + N) on every plane. Fresh sessions may program
    // the extent; restored sessions expose it read-only.
    std::uint32_t published_hbf_blocks_per_plane = 0;
    // Dense logical data that already exist when the measured replay window
    // begins.  The image consumes real HBF data and mapping capacity, but its
    // installation is setup state and therefore does not appear as workload
    // traffic.  It is mutable so later policy-directed writes and GC remain faithful.
    std::uint64_t initial_hbf_logical_first_lpn = 0;
    std::uint64_t initial_hbf_logical_pages = 0;
    // Exact quiescent media/FTL state restored into a fresh controller.
    // Shared ownership keeps app-layer provenance copies cheap even for a
    // large image; HbfController validates and copies the structural state.
    std::shared_ptr<const host::HbfPersistentImage>
        initial_hbf_persistent_image;
    TraceConfig trace;
    // Optional physical-address heatmap over the HBF tier (0 = off). The
    // device records every media program and erase into it; the protocol
    // layer streams per-batch bin deltas for live wear/heat views.
    std::size_t hbf_physical_heatmap_bins = 0;
};

// Persistent execution state for a sequence of causally ordered batches.
// Transaction graphs may depend on completions from earlier batches. A batch
// is complete only after every transaction in its graph has completed. Input
// issue_ns values are batch-relative offsets; the executor rebases them onto
// its persistent completion frontier so independent sessions can consume
// identical canonical timing without sharing device clock state. The
// frontier is the caller's choice per batch (SimulationBatchOptions): it is
// the completion of the named frontier transactions, never earlier than the
// batch's last arrival, so every device still sees nondecreasing arrivals.
class SimulationSession {
public:
    explicit SimulationSession(SimulationSessionConfig config);

    [[nodiscard]] SimulationBatchResult run_batch(
        std::string batch_id,
        const std::vector<SimulationTransaction>& transactions,
        const SimulationBatchOptions& options = {});

    // Read-only setup evidence for the simulation-session ready receipt. This is
    // physical device accounting, not a restatement of the requested image.
    [[nodiscard]] const host::HbfStats* hbf_stats() const {
        return hbf_ ? &hbf_->stats() : nullptr;
    }
    [[nodiscard]] std::uint64_t hbm_application_capacity_bytes() const {
        return hbm_ ? hbm_->application_capacity_bytes() : 0;
    }
    [[nodiscard]] std::uint64_t hbf_buffer_hbm_bytes() const {
        return hbm_ ? hbm_->controller_buffer_bytes() : 0;
    }
    // Explicit transport producers use the controller's exact page owner;
    // they must not assume that an unaligned range divides equally by stack.
    [[nodiscard]] std::size_t hbf_stack_for_logical_page(std::uint64_t lpn) const {
        if (!hbf_) throw SimulationInputError("HBF page routing requires an enabled HBF tier");
        return hbf_->stack_for_logical_page(lpn);
    }
    // O(1) counters for live telemetry. Unlike hbf_stats(), this does not
    // refresh aggregate geometry, wear, or parallel-resource summaries.
    [[nodiscard]] const host::HbfStats* hbf_execution_stats() const {
        return hbf_ ? &hbf_->execution_stats() : nullptr;
    }

    [[nodiscard]] SimulationDeviceSnapshot device_snapshot() const;
    // Observe state at the completed frontier. Only already-completed lazy
    // callbacks are materialized; no drain, new IO or time advance is issued.
    [[nodiscard]] SimulationObservationResult observe_completed_state();
    [[nodiscard]] const AddressHeatmap* hbf_heatmap() const {
        return hbf_heatmap_.get();
    }
    [[nodiscard]] std::vector<host::HbfBlockProfileBin> hbf_block_profile(
        std::size_t bin_count) const {
        return hbf_ ? hbf_->block_profile(bin_count) :
            std::vector<host::HbfBlockProfileBin>{};
    }
    [[nodiscard]] std::vector<std::uint32_t> hbf_block_erase_counts() const {
        return hbf_ ? hbf_->block_erase_counts() :
            std::vector<std::uint32_t>{};
    }
    [[nodiscard]] std::string hbf_wear_snapshot_json() const;
    [[nodiscard]] std::string hbf_mapping_snapshot_json() const;
    [[nodiscard]] std::pair<std::uint64_t,std::uint64_t> hbf_initial_image_media_pages() const {
        return hbf_ ? hbf_->initial_image_media_pages() : std::pair<std::uint64_t,std::uint64_t>{};
    }
    [[nodiscard]] PhysicalCompletion hbf_object_operation(std::string_view action,
        std::string id, std::uint64_t object_id, std::uint64_t first_lpn, std::uint64_t page_count);
    // Host commands are explicit IO barriers outside the transaction census.
    [[nodiscard]] PhysicalCompletion hbf_zone_operation(std::string_view command,
        std::string id, std::uint32_t stack, std::uint32_t channel,
        std::uint64_t zone, std::uint64_t argument = 0);
    // Fragment arrival times are offsets from the current completed frontier.
    [[nodiscard]] std::vector<host::HbfController::ChannelWriteCompletion> hbf_channel_write_batch(
        std::vector<PhysicalRequest> requests);
    [[nodiscard]] SimulationInvalidationResult invalidate_hbf_pages(
        std::string id, std::uint64_t first_lpn, std::uint64_t page_count);
    // Persist all pending mutable HBF state without ending the session. The
    // completion frontier advances through the checkpoint, so every later
    // batch is causally ordered after the persisted image.
    [[nodiscard]] SimulationCheckpointResult checkpoint_pending(
        std::string checkpoint_id);
    [[nodiscard]] host::HbfPersistentImage persistent_hbf_image() const;
    // Terminate the session at the boundary after the previous completed
    // protocol command. Lazy callbacks are materialized only through the
    // already-completed frontier; this deliberately performs no HBF drain,
    // future commit advancement, mapping writeback, or persistent-image
    // export. The caller must recover from an already published artifact.
    [[nodiscard]] SimulationCrashResult inject_crash(std::string crash_id);
    [[nodiscard]] SimulationDrainResult drain_pending();

    [[nodiscard]] const auto& completion_stats() const {
        return completion_by_target_;
    }

    [[nodiscard]] std::size_t completed_batches() const {
        return next_sequence_;
    }

    [[nodiscard]] std::size_t completed_checkpoints() const {
        return next_checkpoint_sequence_;
    }

    [[nodiscard]] double completed_frontier_ns() const {
        return completed_frontier_ns_;
    }

    // Transaction ids currently resolvable as cross-batch dependencies.
    [[nodiscard]] std::size_t resolvable_dependency_ids() const;

private:
    using FinishById = std::unordered_map<std::string, double>;

    SimulationSessionConfig config_;
    std::unique_ptr<hbm::HbmDevice> hbm_;
    std::unique_ptr<host::HbfController> hbf_;
    std::uint64_t hbf_logical_capacity_pages_ = 0;
    std::unique_ptr<AddressHeatmap> hbf_heatmap_;
    std::unique_ptr<external::ExternalBackingDevice> external_;
    std::unique_ptr<external::ExternalBackingDevice> host_dram_;
    std::vector<BaseDieLink> base_die_links_;
    std::vector<BaseDieLink> hbf_external_direct_links_;
    // Completion times of the last kDependencyWindowBatches batches (oldest
    // first) plus the ids the caller explicitly retained.
    std::deque<FinishById> recent_batch_finishes_;
    FinishById retained_finishes_;
    std::unordered_set<std::string> batch_ids_;
    std::unordered_set<std::string> checkpoint_ids_;
    std::size_t next_sequence_ = 0;
    std::size_t next_checkpoint_sequence_ = 0;
    // Blocking frontier: origin of the next batch.
    double completed_frontier_ns_ = 0.0;
    // Completion of all issued work, detached transactions included; drains
    // and checkpoints start no earlier than this.
    double issued_work_frontier_ns_ = 0.0;
    bool failed_ = false;
    bool drained_ = false;
    std::array<
        std::array<SimulationCompletionStats, kSimulationOperationCount>,
        kSimulationTargetCount> completion_by_target_{};

    void validate_transaction(const SimulationTransaction& transaction) const;
    [[nodiscard]] const double* find_completed_finish(
        const std::string& id) const;
    [[nodiscard]] SimulationDeviceSnapshot device_counters() const;
    [[nodiscard]] PhysicalCompletion issue_non_hbm_memory(
        const SimulationTransaction& transaction,
        double ready_ns);
    [[nodiscard]] SimulationDrainResult persist_pending(
        std::string completion_id,
        bool end_session);
};

} // namespace hbfsim::physical
