#pragma once

#include "physical/hybrid/composition_common.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hbfsim::physical::hybrid {

// Admission is deliberately defined only over the observed address/op stream.
// Neither policy is allowed to inspect semantic kind, label, phase, layer, or
// compute annotations.
enum class BehavioralAdmissionPolicy {
    // Conventional demand-fill baseline: every nonresident page is promoted.
    AlwaysAdmit,
    // Bounded ghost history rejects first-touch scan traffic. A page is
    // promoted only after promotion_threshold nonresident observations.
    ReuseFiltered,
};

[[nodiscard]] const char* to_string(BehavioralAdmissionPolicy policy);

struct BehavioralTieringConfig {
    hbm::HbmConfig hbm;
    hbf::HbfConfig hbf;
    BaseDieLinkConfig base_die_link;
    BehavioralAdmissionPolicy admission_policy =
        BehavioralAdmissionPolicy::ReuseFiltered;

    // Physical HBM capacity owned by this controller. It must be a positive,
    // page-aligned prefix of the configured HBM device.
    std::uint64_t hbm_tier_bytes = 0;
    // ReuseFiltered promotes on this bounded-history observation count.
    // Values below two would collapse the policy into AlwaysAdmit.
    std::uint32_t promotion_threshold = 2;
    // Maximum nonresident page identities retained by the controller.
    // Metadata is allocated only for identities actually observed.
    std::uint64_t history_capacity_pages = 1'048'576;

    // Completion-order credit limit for foreground page transactions. Zero is
    // open loop. Ready parents share credits round-robin, matching the direct
    // composition contract; a promotion keeps its credit through fill/install.
    std::size_t max_outstanding_requests = 0;
    std::size_t address_heatmap_bins = 0;
    TraceConfig trace;
    bool retain_completions = true;
    bool retain_decisions = false;
    // Optional population contract for prefix/scaling experiments. It affects
    // only initial backing contents, never online placement decisions.
    const std::vector<MemoryRequest>* initial_image_requests = nullptr;
};

enum class BehavioralDecisionAction {
    HbmHit,
    HbfBypass,
    Promote,
};

[[nodiscard]] const char* to_string(BehavioralDecisionAction action);

struct BehavioralPlacementDecision {
    std::uint64_t observation = 0;
    std::size_t request_index = 0;
    std::size_t transaction_index = 0;
    std::uint64_t logical_page = 0;
    Op op = Op::Read;
    BehavioralDecisionAction action = BehavioralDecisionAction::HbfBypass;
    std::uint32_t history_observations = 0;
    std::optional<std::uint64_t> hbm_slot;
    std::optional<std::uint64_t> evicted_page;
    bool evicted_page_dirty = false;
};

struct BehavioralTieringStats {
    BehavioralAdmissionPolicy admission_policy =
        BehavioralAdmissionPolicy::ReuseFiltered;
    // A hard contract, not a workload observation. It remains false unless
    // the implementation has violated its public policy boundary.
    bool semantic_inputs_consumed = false;
    std::uint64_t hbm_tier_pages = 0;
    std::uint64_t hbm_tier_bytes = 0;
    std::uint32_t promotion_threshold = 0;
    std::uint64_t history_capacity_pages = 0;

    std::uint64_t page_observations = 0;
    std::uint64_t hbm_hits = 0;
    std::uint64_t hbf_bypasses = 0;
    std::uint64_t cold_misses = 0;
    std::uint64_t history_hits = 0;
    std::uint64_t promotions = 0;
    std::uint64_t promotions_with_backing_fill = 0;
    std::uint64_t promotions_without_backing_fill = 0;
    std::uint64_t clean_evictions = 0;
    std::uint64_t dirty_evictions = 0;
    std::uint64_t history_evictions = 0;

    std::uint64_t hbm_foreground_bytes = 0;
    std::uint64_t hbf_bypass_bytes = 0;
    std::uint64_t backing_fill_pages = 0;
    std::uint64_t backing_fill_bytes = 0;
    std::uint64_t hbm_install_pages = 0;
    std::uint64_t hbm_install_bytes = 0;
    std::uint64_t dirty_writeback_pages = 0;
    std::uint64_t dirty_writeback_bytes = 0;
    std::uint64_t drain_writeback_pages = 0;
    std::uint64_t drain_writeback_bytes = 0;

    std::uint64_t peak_resident_pages = 0;
    std::uint64_t final_resident_pages = 0;
    std::uint64_t final_dirty_pages = 0;
    std::uint64_t peak_history_pages = 0;
    std::uint64_t peak_installing_pages = 0;
    std::uint64_t capacity_stalled_promotions = 0;
    double capacity_stall_work_ns = 0.0;
    std::uint64_t transition_waited_transactions = 0;
    double transition_wait_work_ns = 0.0;
    double transition_max_wait_ns = 0.0;

    // Stable FNV-1a fingerprint over the complete causal decision ledger:
    // action, history evidence, HBM slot, and eviction choice. It
    // intentionally excludes semantic annotations and physical timing.
    std::uint64_t decision_fingerprint = 1469598103934665603ull;
};

struct BehavioralTieringRunResult {
    CompositionRunResult composition;
    BehavioralTieringStats placement;
    std::vector<BehavioralPlacementDecision> decisions;
};

[[nodiscard]] BehavioralTieringRunResult
run_behavioral_tiering_composition(const BehavioralTieringConfig& config,
                                   const std::vector<MemoryRequest>& requests);

} // namespace hbfsim::physical::hybrid
