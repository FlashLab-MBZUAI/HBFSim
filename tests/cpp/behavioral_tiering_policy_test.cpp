#include "policies/reference/behavioral_tiering_policy.hpp"
#include "policies/reference/direct_policy.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using hbfsim::physical::Op;
using hbfsim::policy::BehavioralAdmissionPolicy;
using hbfsim::policy::BehavioralDecisionAction;
using hbfsim::policy::BehavioralTieringConfig;
using hbfsim::policy::DirectRunKnobs;
using hbfsim::policy::MemoryRequest;
using hbfsim::policy::all_hbf_policy;
using hbfsim::policy::run_direct_policy;
using hbfsim::policy::run_behavioral_tiering_policy;
using hbfsim::policy::SemanticKind;

constexpr std::uint64_t kPage = 4096;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void require_throws(Fn&& fn, const std::string& message) {
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, message);
}

BehavioralTieringConfig base_config(std::uint64_t hbm_pages = 2) {
    BehavioralTieringConfig config;
    // The application tier stays tiny; physical HBM also owns HBF buffers.
    config.hbm.device.capacity_bytes = 16 * 1024 * 1024;
    config.hbm.device.stacks = 1;
    config.hbm.device.channels_per_stack = 2;
    config.hbm.device.pseudo_channels_per_channel = 1;

    config.hbf.device.stacks = 2;
    config.hbf.device.channels_per_stack = 1;
    config.hbf.device.dies_per_channel = 1;
    config.hbf.device.planes_per_die = 2;
    config.hbf.device.blocks_per_plane = 16;
    config.hbf.device.pages_per_block = 16;
    config.hbf.device.page_size_bytes = kPage;
    config.hbf.device.oob_bytes_per_page = 224;
    config.hbf.host.mapping_entries_per_page = 64;

    config.hbf.host.write_coalescing_enabled = false;
    config.hbf.host.write_buffer_completion_requires_flush = true;
    config.hbf.host.gc_low_watermark_pages = 0;
    // Automatic GC requires a reserve the foreground can never open (one
    // block under full-resident mapping, two under cached); two is valid
    // for either mapping mode and leaves 16 - 2 - 3 = 11 data blocks per
    // plane for these small workloads.
    config.hbf.host.gc_reserved_free_blocks_per_plane = 2;

    config.base_die_link.read_bandwidth_GBps = 64.0;
    config.base_die_link.write_bandwidth_GBps = 64.0;
    config.base_die_link.latency_ns = 20.0;
    config.hbm_tier_bytes = hbm_pages * kPage;
    config.history_capacity_pages = 16;
    config.promotion_threshold = 2;
    config.max_outstanding_requests = 8;
    config.retain_completions = true;
    config.retain_decisions = true;
    config.address_heatmap_bins = 16;
    return config;
}

MemoryRequest request(std::size_t index, std::uint64_t page, Op op = Op::Read,
                      double arrival_ns = 0.0) {
    return MemoryRequest{
        .id = "op" + std::to_string(index),
        .op = op,
        .addr = page * kPage,
        .bytes = kPage,
        .arrival_ns = arrival_ns,
        .index = index,
    };
}

void test_scan_is_bypassed() {
    auto config = base_config();
    std::vector<MemoryRequest> requests;
    for (std::size_t page = 0; page < 4; ++page) {
        requests.push_back(request(page, page, Op::Read, page));
    }
    const auto run = run_behavioral_tiering_policy(config, requests);
    require(run.placement.page_observations == 4 &&
                run.placement.cold_misses == 4 &&
                run.placement.hbf_bypasses == 4 &&
                run.placement.promotions == 0 && run.placement.hbm_hits == 0,
            "reuse filter did not reject a one-pass scan");
    require(run.composition.hbf_user_accesses == 4 &&
                run.composition.hbm_user_accesses == 0 &&
                run.composition.hbm_background_accesses == 0,
            "one-pass scan exercised the wrong physical tier");
    require(run.decisions.size() == 4 &&
                std::all_of(run.decisions.begin(), run.decisions.end(),
                            [](const auto& decision) {
                                return decision.action ==
                                       BehavioralDecisionAction::HbfBypass;
                            }),
            "one-pass scan decision ledger drifted");
}

void test_reuse_promotes_then_hits() {
    auto config = base_config();
    std::vector<MemoryRequest> requests;
    for (std::size_t index = 0; index < 4; ++index) {
        requests.push_back(request(index, 0, Op::Read, 0.0));
    }
    const auto run = run_behavioral_tiering_policy(config, requests);
    require(run.placement.hbf_bypasses == 1 && run.placement.promotions == 1 &&
                run.placement.promotions_with_backing_fill == 1 &&
                run.placement.hbm_hits == 2,
            "reused page did not follow bypass/promote/hit lifecycle");
    require(run.placement.backing_fill_pages == 1 &&
                run.placement.hbm_install_pages == 1 &&
                run.composition.base_die_link_stats.read_bytes == kPage,
            "promotion fill traffic does not conserve one page");
    require(run.composition.hbf_user_accesses == 1 &&
                run.composition.hbf_background_accesses == 2 &&
                run.composition.hbm_user_accesses == 2 &&
                run.composition.hbm_background_accesses == 1,
            "promotion foreground/background access census drifted");
    require(run.placement.transition_waited_transactions >= 1,
            "causal page-decision chain was not fenced");
}

void test_semantic_annotations_are_inert() {
    auto config = base_config();
    std::vector<MemoryRequest> plain{
        request(0, 2, Op::Read, 0.0),
        request(1, 2, Op::Read, 1.0),
        request(2, 2, Op::Write, 2.0),
        request(3, 3, Op::Read, 3.0),
    };
    auto annotated = plain;
    for (std::size_t index = 0; index < annotated.size(); ++index) {
        annotated[index].kind =
            index % 2 == 0 ? SemanticKind::ModelWeights : SemanticKind::Scratch;
        annotated[index].label = "semantic-" + std::to_string(index);
        annotated[index].phase = 100 + index;
        annotated[index].layer = 1000 + index;
        annotated[index].compute_ns = 17.0 + index;
    }
    const auto first = run_behavioral_tiering_policy(config, plain);
    const auto second = run_behavioral_tiering_policy(config, annotated);
    require(!first.placement.semantic_inputs_consumed &&
                !second.placement.semantic_inputs_consumed,
            "behavioral controller claims semantic input consumption");
    require(first.placement.decision_fingerprint ==
                    second.placement.decision_fingerprint &&
                first.placement.hbm_hits == second.placement.hbm_hits &&
                first.placement.hbf_bypasses == second.placement.hbf_bypasses &&
                first.placement.promotions == second.placement.promotions,
            "semantic annotations changed behavior-only placement");
    require(first.composition.hbm_stats.read_bytes ==
                    second.composition.hbm_stats.read_bytes &&
                first.composition.hbm_stats.write_bytes ==
                    second.composition.hbm_stats.write_bytes &&
                first.composition.hbf_stats.logical_read_bytes ==
                    second.composition.hbf_stats.logical_read_bytes &&
                first.composition.hbf_stats.logical_write_bytes ==
                    second.composition.hbf_stats.logical_write_bytes,
            "semantic annotations changed physical traffic");
}

void test_time_translation_and_future_suffix_are_causal() {
    auto config = base_config(2);
    std::vector<MemoryRequest> prefix{
        request(0, 0, Op::Read, 0.0),
        request(1, 0, Op::Read, 1'000'000.0),
        request(2, 1, Op::Write, 2'000'000.0),
        request(3, 1, Op::Read, 3'000'000.0),
        request(4, 0, Op::Read, 4'000'000.0),
    };
    auto shifted = prefix;
    for (auto& item : shifted) {
        item.arrival_ns += 10'000'000.0;
    }
    const auto baseline = run_behavioral_tiering_policy(config, prefix);
    const auto translated = run_behavioral_tiering_policy(config, shifted);
    require(baseline.placement.decision_fingerprint ==
                    translated.placement.decision_fingerprint &&
                baseline.placement.hbm_hits == translated.placement.hbm_hits &&
                baseline.placement.hbf_bypasses ==
                    translated.placement.hbf_bypasses &&
                baseline.placement.promotions ==
                    translated.placement.promotions &&
                baseline.placement.dirty_writeback_pages ==
                    translated.placement.dirty_writeback_pages,
            "absolute time translation changed placement state");

    auto extended = prefix;
    extended.push_back(request(5, 7, Op::Read, 5'000'000.0));
    extended.push_back(request(6, 7, Op::Read, 6'000'000.0));
    extended.push_back(request(7, 8, Op::Write, 7'000'000.0));
    const auto with_suffix =
        run_behavioral_tiering_policy(config, extended);
    require(with_suffix.decisions.size() >= baseline.decisions.size(),
            "future suffix lost prefix decisions");
    for (std::size_t index = 0; index < baseline.decisions.size(); ++index) {
        const auto& expected = baseline.decisions[index];
        const auto& actual = with_suffix.decisions[index];
        require(
            expected.observation == actual.observation &&
                expected.request_index == actual.request_index &&
                expected.transaction_index == actual.transaction_index &&
                expected.logical_page == actual.logical_page &&
                expected.op == actual.op && expected.action == actual.action &&
                expected.history_observations == actual.history_observations &&
                expected.hbm_slot == actual.hbm_slot &&
                expected.evicted_page == actual.evicted_page &&
                expected.evicted_page_dirty == actual.evicted_page_dirty,
            "future suffix changed an earlier placement decision");
    }
}

void test_phase_dependency_is_execution_only() {
    auto config = base_config(2);
    config.max_outstanding_requests = 2;
    std::vector<MemoryRequest> plain{
        request(0, 0),
        request(1, 1),
        request(2, 2),
        request(3, 3),
    };
    auto phased = plain;
    phased[0].phase = 0;
    phased[1].phase = 0;
    phased[2].phase = 1;
    phased[3].phase = 1;

    const auto unphased =
        run_behavioral_tiering_policy(config, plain);
    const auto behavioral =
        run_behavioral_tiering_policy(config, phased);
    require(
        behavioral.placement.decision_fingerprint ==
                unphased.placement.decision_fingerprint &&
            behavioral.placement.hbf_bypasses == plain.size() &&
            behavioral.placement.promotions == 0,
        "phase dependency leaked into behavior-only placement");
    require(
        behavioral.composition.phase_barriers == 1 &&
            behavioral.composition.phase_dependency_waited_ops == 2 &&
            behavioral.composition.phase_dependency_wait_work_ns > 0.0 &&
            behavioral.composition.phase_dependency_max_wait_ns > 0.0 &&
            behavioral.composition.last_offered_arrival_ns >
                behavioral.composition.first_offered_arrival_ns,
        "behavioral execution ignored an explicit phase barrier");
    require(
        behavioral.composition.source_latencies_ns[2] >
                behavioral.composition.offered_latencies_ns[2] &&
            behavioral.composition.source_latencies_ns[3] >
                behavioral.composition.offered_latencies_ns[3],
        "behavioral latency views lost phase-dependency wait");

    DirectRunKnobs knobs;
    knobs.max_outstanding_requests = config.max_outstanding_requests;
    knobs.retain_completions = true;
    const auto direct = run_direct_policy(
        all_hbf_policy(), config.hbm, config.hbf, phased, knobs);
    require(
        direct.phase_barriers == behavioral.composition.phase_barriers &&
            direct.phase_dependency_waited_ops ==
                behavioral.composition.phase_dependency_waited_ops &&
            direct.first_offered_arrival_ns ==
                behavioral.composition.first_offered_arrival_ns &&
            direct.last_offered_arrival_ns ==
                behavioral.composition.last_offered_arrival_ns &&
            direct.user_finish_ns == behavioral.composition.user_finish_ns,
        "cold-bypass phase scheduling diverged from the direct all-hbf "
        "front-end contract");
}

void test_cold_bypass_matches_direct_parent_fairness() {
    auto config = base_config(2);
    config.max_outstanding_requests = 2;
    auto first = request(0, 0);
    first.bytes = 4 * kPage;
    auto second = request(1, 64);
    second.bytes = 4 * kPage;
    const std::vector<MemoryRequest> requests{first, second};

    const auto behavioral =
        run_behavioral_tiering_policy(config, requests);
    DirectRunKnobs knobs;
    knobs.max_outstanding_requests = config.max_outstanding_requests;
    knobs.retain_completions = true;
    const auto direct = run_direct_policy(
        all_hbf_policy(), config.hbm, config.hbf, requests, knobs);
    require(behavioral.placement.hbf_bypasses == 8 &&
                behavioral.placement.promotions == 0,
            "cold-bypass parent unexpectedly changed placement");
    require(direct.hbf_stats.logical_read_bytes ==
                behavioral.composition.hbf_stats.logical_read_bytes,
            "cold-bypass parent changed direct all-hbf traffic");
    require(direct.user_finish_ns == behavioral.composition.user_finish_ns,
            "cold-bypass parent changed direct all-hbf user finish");
    require(
        direct.hbf_stats.stage_work.scheduler_queue_wait_ns ==
            behavioral.composition.hbf_stats.stage_work.scheduler_queue_wait_ns,
        "cold-bypass parent changed direct all-hbf scheduler wait");
}

void test_policy_is_invariant_to_window_and_device_timing() {
    auto sequential_config = base_config(2);
    sequential_config.max_outstanding_requests = 1;
    const std::vector<std::pair<std::uint64_t, Op>> stream{
        {0, Op::Read}, {1, Op::Write}, {2, Op::Read},
        {0, Op::Read}, {1, Op::Read},  {2, Op::Write},
        {3, Op::Read}, {0, Op::Read},  {3, Op::Read},
    };
    std::vector<MemoryRequest> spaced;
    std::vector<MemoryRequest> burst;
    for (std::size_t index = 0; index < stream.size(); ++index) {
        spaced.push_back(request(index, stream[index].first,
                                 stream[index].second, index * 1'000'000.0));
        burst.push_back(request(index, stream[index].first,
                                stream[index].second,
                                static_cast<double>(index)));
    }
    auto burst_config = sequential_config;
    burst_config.max_outstanding_requests = 8;
    // Change physical response timing too. The policy has no timing input, so
    // neither this nor the front-end window may alter its action stream.
    burst_config.hbf.device.t_read_page_ns *= 2.0;
    burst_config.hbf.device.ecc_decode_latency_ns *= 1.5;

    const auto sequential =
        run_behavioral_tiering_policy(sequential_config, spaced);
    const auto concurrent =
        run_behavioral_tiering_policy(burst_config, burst);
    require(sequential.placement.decision_fingerprint ==
                    concurrent.placement.decision_fingerprint &&
                sequential.placement.hbm_hits ==
                    concurrent.placement.hbm_hits &&
                sequential.placement.hbf_bypasses ==
                    concurrent.placement.hbf_bypasses &&
                sequential.placement.promotions ==
                    concurrent.placement.promotions &&
                sequential.placement.clean_evictions ==
                    concurrent.placement.clean_evictions &&
                sequential.placement.dirty_evictions ==
                    concurrent.placement.dirty_evictions &&
                sequential.placement.dirty_writeback_pages ==
                    concurrent.placement.dirty_writeback_pages &&
                sequential.decisions.size() == concurrent.decisions.size(),
            "window or device timing changed behavior-only placement");
    for (std::size_t index = 0; index < sequential.decisions.size(); ++index) {
        const auto& expected = sequential.decisions[index];
        const auto& actual = concurrent.decisions[index];
        require(
            expected.request_index == actual.request_index &&
                expected.logical_page == actual.logical_page &&
                expected.op == actual.op && expected.action == actual.action &&
                expected.history_observations == actual.history_observations &&
                expected.hbm_slot == actual.hbm_slot &&
                expected.evicted_page == actual.evicted_page &&
                expected.evicted_page_dirty == actual.evicted_page_dirty,
            "window or timing changed a retained placement decision");
    }
}

void test_page_identity_renaming_preserves_policy_shape() {
    auto config = base_config(2);
    const std::vector<MemoryRequest> original{
        request(0, 0, Op::Read, 0.0),
        request(1, 0, Op::Read, 1'000'000.0),
        request(2, 1, Op::Write, 2'000'000.0),
        request(3, 1, Op::Read, 3'000'000.0),
        request(4, 2, Op::Read, 4'000'000.0),
        request(5, 2, Op::Read, 5'000'000.0),
        request(6, 0, Op::Read, 6'000'000.0),
    };
    auto renamed = original;
    for (auto& item : renamed) {
        const auto page = item.addr / kPage;
        const auto mapped = page == 0 ? 9 : page == 1 ? 4 : 12;
        item.addr = mapped * kPage;
    }
    const auto first = run_behavioral_tiering_policy(config, original);
    const auto second = run_behavioral_tiering_policy(config, renamed);
    require(first.placement.hbm_hits == second.placement.hbm_hits &&
                first.placement.hbf_bypasses == second.placement.hbf_bypasses &&
                first.placement.promotions == second.placement.promotions &&
                first.placement.clean_evictions ==
                    second.placement.clean_evictions &&
                first.placement.dirty_evictions ==
                    second.placement.dirty_evictions &&
                first.placement.dirty_writeback_pages ==
                    second.placement.dirty_writeback_pages &&
                first.decisions.size() == second.decisions.size(),
            "bijective page renaming changed policy-level counts");
    for (std::size_t index = 0; index < first.decisions.size(); ++index) {
        require(first.decisions[index].action ==
                        second.decisions[index].action &&
                    first.decisions[index].history_observations ==
                        second.decisions[index].history_observations &&
                    first.decisions[index].hbm_slot ==
                        second.decisions[index].hbm_slot &&
                    first.decisions[index].evicted_page_dirty ==
                        second.decisions[index].evicted_page_dirty,
                "bijective page renaming changed decision shape");
    }
}

void test_dirty_eviction_and_drain_conserve() {
    auto config = base_config(1);
    config.admission_policy = BehavioralAdmissionPolicy::AlwaysAdmit;
    const std::vector<MemoryRequest> requests{
        request(0, 0, Op::Write, 0.0),
        request(1, 1, Op::Write, 0.0),
    };
    const auto run = run_behavioral_tiering_policy(config, requests);
    require(run.placement.promotions == 2 &&
                run.placement.promotions_without_backing_fill == 2 &&
                run.placement.dirty_evictions == 1 &&
                run.placement.clean_evictions == 0,
            "one-slot dirty lifecycle drifted");
    require(run.placement.dirty_writeback_pages == 2 &&
                run.placement.drain_writeback_pages == 1 &&
                run.placement.dirty_writeback_bytes == 2 * kPage &&
                run.composition.base_die_link_stats.write_bytes == 2 * kPage,
            "dirty eviction plus drain does not conserve bytes");
    require(run.composition.hbf_stats.logical_write_bytes == 2 * kPage &&
                run.composition.hbm_user_accesses == 2 &&
                run.composition.hbm_background_accesses == 2 &&
                run.placement.final_resident_pages == 1 &&
                run.placement.final_dirty_pages == 0,
            "dirty lifecycle final state is inconsistent");
    require(run.decisions.size() == 2 && run.decisions[1].evicted_page == 0 &&
                run.decisions[1].evicted_page_dirty,
            "dirty victim is missing from the decision ledger");
}

void test_pending_eviction_fences_repromotion() {
    auto config = base_config(1);
    config.admission_policy = BehavioralAdmissionPolicy::AlwaysAdmit;
    config.max_outstanding_requests = 3;
    const std::vector<MemoryRequest> requests{
        request(0, 0, Op::Write, 0.0),
        request(1, 1, Op::Read, 0.0),
        request(2, 0, Op::Read, 0.0),
    };
    const auto run = run_behavioral_tiering_policy(config, requests);
    require(
            run.placement.promotions == 3 &&
            run.placement.dirty_evictions == 1 &&
            run.placement.clean_evictions == 1 &&
            run.placement.transition_waited_transactions >= 1 &&
            run.placement.final_resident_pages == 1 &&
            run.placement.final_dirty_pages == 0,
        "access to a pending eviction victim was not causally fenced");
    require(
        run.decisions.size() == 3 &&
            run.decisions[1].evicted_page == 0 &&
            run.decisions[1].evicted_page_dirty &&
            run.decisions[2].evicted_page == 1 &&
            !run.decisions[2].evicted_page_dirty,
        "pending-eviction repro-motion changed the planned victim chain");
}

void test_planned_hit_waits_for_earlier_parent_promotion() {
    auto config = base_config(8);
    config.admission_policy = BehavioralAdmissionPolicy::AlwaysAdmit;
    config.max_outstanding_requests = 2;
    auto multi_page = request(0, 0, Op::Read, 0.0);
    multi_page.bytes = 3 * kPage;
    const std::vector<MemoryRequest> requests{
        multi_page,
        request(1, 2, Op::Read, 0.0),
    };
    const auto run = run_behavioral_tiering_policy(config, requests);
    require(
        run.placement.promotions == 3 &&
            run.placement.hbm_hits == 1 &&
            run.placement.transition_waited_transactions >= 1 &&
            run.composition.ops == 2 &&
            run.placement.final_resident_pages == 3,
        "round-robin hit overtook an earlier-parent promotion");
    require(
        run.decisions.size() == 4 &&
            run.decisions[2].logical_page == 2 &&
            run.decisions[2].action ==
                BehavioralDecisionAction::Promote &&
            run.decisions[3].logical_page == 2 &&
            run.decisions[3].action ==
                BehavioralDecisionAction::HbmHit,
        "multi-page promotion/hit causal ledger drifted");
}

void test_repromotion_waits_for_intervening_eviction() {
    auto config = base_config(2);
    config.admission_policy = BehavioralAdmissionPolicy::AlwaysAdmit;
    config.max_outstanding_requests = 4;
    const std::vector<MemoryRequest> requests{
        request(0, 0, Op::Read, 0.0),
        request(1, 1, Op::Read, 0.0),
        request(2, 2, Op::Read, 0.0),
        request(3, 0, Op::Read, 0.0),
    };
    const auto run = run_behavioral_tiering_policy(config, requests);
    require(
        run.placement.promotions == 4 &&
            run.placement.clean_evictions == 2 &&
            run.placement.transition_waited_transactions >= 1 &&
            run.composition.ops == requests.size() &&
            run.placement.final_resident_pages == 2,
        "re-promotion overtook the replacement that evicted its old copy");
    require(
        run.decisions.size() == 4 &&
            run.decisions[2].evicted_page == 0 &&
            run.decisions[3].logical_page == 0 &&
            run.decisions[3].evicted_page == 1,
        "intervening-eviction decision chain drifted");
}

void test_replacement_is_true_lru() {
    auto config = base_config(2);
    config.admission_policy = BehavioralAdmissionPolicy::AlwaysAdmit;
    const std::vector<MemoryRequest> requests{
        request(0, 0, Op::Read, 0.0),
        request(1, 1, Op::Read, 1'000'000.0),
        request(2, 0, Op::Read, 2'000'000.0),
        request(3, 2, Op::Read, 3'000'000.0),
    };
    const auto run = run_behavioral_tiering_policy(config, requests);
    require(
        run.decisions.size() == requests.size() &&
            run.decisions[3].action == BehavioralDecisionAction::Promote &&
            run.decisions[3].evicted_page == 1 &&
            !run.decisions[3].evicted_page_dirty,
        "behavioral replacement did not evict the least-recently used page");
    require(run.placement.clean_evictions == 1 &&
                run.placement.dirty_evictions == 0 &&
                run.placement.final_resident_pages == 2,
            "LRU replacement lifecycle accounting drifted");
}

// A tier of one million pages filled by distinct clean pages, then a burst
// of further promotions that must each evict the least-recently observed
// resident. The former linear victim scan made this O(promotions x
// residents); the intrusive LRU list keeps the whole run in seconds.
void test_million_page_tier_evicts_in_constant_time() {
    constexpr std::uint64_t tier_pages = 1'048'576;
    constexpr std::uint64_t overflow_pages = 8192;
    auto config = base_config();
    config.admission_policy = BehavioralAdmissionPolicy::AlwaysAdmit;
    config.hbm.device.stacks = 1;
    config.hbm.device.channels_per_stack = 8;
    config.hbm.device.pseudo_channels_per_channel = 2;
    // 2 stacks x 1 channel x 1 die x 2 planes x 256 pages/block x 4 KiB
    // = 4 MiB per block-per-plane step: 1280 blocks leave room for the
    // 4.03 GiB of read-before-write pages the run prepopulates plus their
    // mapping checkpoints and the GC reserve.
    config.hbf.device.pages_per_block = 256;
    config.hbf.device.blocks_per_plane = 1280;
    config.hbm_tier_bytes = tier_pages * kPage;
    config.hbm.device.capacity_bytes = config.hbm_tier_bytes +
        hbfsim::host::HbfController(config.hbf).config().host.ctrl_dram_bytes;
    config.history_capacity_pages = 1;
    config.max_outstanding_requests = 64;
    config.retain_completions = false;
    config.retain_decisions = false;
    config.address_heatmap_bins = 0;
    std::vector<MemoryRequest> requests;
    requests.reserve(tier_pages + overflow_pages);
    for (std::uint64_t page = 0; page < tier_pages + overflow_pages; ++page) {
        requests.push_back(request(page, page, Op::Read, 0.0));
    }
    const auto run = run_behavioral_tiering_policy(config, requests);
    require(run.placement.promotions == tier_pages + overflow_pages &&
                run.placement.clean_evictions == overflow_pages &&
                run.placement.dirty_evictions == 0 &&
                run.placement.peak_resident_pages == tier_pages &&
                run.placement.final_resident_pages == tier_pages,
            "million-page tier did not fill and then evict once per overflow");
    require(run.composition.hbm_user_accesses == 0 &&
                run.composition.hbf_user_accesses == 0 &&
                run.composition.hbm_background_accesses ==
                    tier_pages + overflow_pages,
            "million-page promotions did not install exactly one page each");
}

void test_policy_validation_fails_closed() {
    auto config = base_config();
    config.promotion_threshold = 1;
    require_throws(
        [&] {
            (void)run_behavioral_tiering_policy(
                config, std::vector<MemoryRequest>{request(0, 0)});
        },
        "reuse filter accepted a one-touch promotion threshold");
    config = base_config();
    config.hbm_tier_bytes = kPage + 1;
    require_throws(
        [&] {
            (void)run_behavioral_tiering_policy(
                config, std::vector<MemoryRequest>{request(0, 0)});
        },
        "behavioral tier accepted a partial physical page");
}

} // namespace

int main() {
    try {
        test_scan_is_bypassed();
        test_reuse_promotes_then_hits();
        test_semantic_annotations_are_inert();
        test_time_translation_and_future_suffix_are_causal();
        test_phase_dependency_is_execution_only();
        test_cold_bypass_matches_direct_parent_fairness();
        test_policy_is_invariant_to_window_and_device_timing();
        test_page_identity_renaming_preserves_policy_shape();
        test_dirty_eviction_and_drain_conserve();
        test_pending_eviction_fences_repromotion();
        test_planned_hit_waits_for_earlier_parent_promotion();
        test_repromotion_waits_for_intervening_eviction();
        test_replacement_is_true_lru();
        test_million_page_tier_evicts_in_constant_time();
        test_policy_validation_fails_closed();
        std::cout << "behavioral tiering composition tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "behavioral tiering composition test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
