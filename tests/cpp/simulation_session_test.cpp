#include "physical/simulation_session.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using hbfsim::physical::Op;
using hbfsim::physical::SimulationSession;
using hbfsim::physical::SimulationSessionConfig;
using hbfsim::physical::SimulationTarget;
using hbfsim::physical::SimulationTransaction;

constexpr std::uint64_t kPage = 4096;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(message);
}

SimulationSessionConfig base_config(bool enable_hbf = true) {
    SimulationSessionConfig config;
    config.enable_hbm = true;
    config.enable_hbf = enable_hbf;
    config.hbm.device.capacity_bytes = 32 * kPage;
    config.hbm.device.stacks = 1;
    config.hbm.device.channels_per_stack = 2;
    config.hbm.device.pseudo_channels_per_channel = 1;
    config.hbm.device.bank_groups_per_pseudo_channel = 1;
    config.hbm.device.banks_per_group = 2;

    config.hbf.device.stacks = 2;
    config.hbf.device.channels_per_stack = 1;
    config.hbf.device.dies_per_channel = 1;
    config.hbf.device.planes_per_die = 2;
    config.hbf.device.blocks_per_plane = 8;
    config.hbf.device.pages_per_block = 8;
    config.hbf.device.page_size_bytes = kPage;
    config.hbf.device.oob_bytes_per_page = 224;
    config.hbf.host.mapping_entries_per_page = 64;

    config.hbf.host.write_coalescing_enabled = false;
    config.hbf.host.write_buffer_completion_requires_flush = true;
    config.hbf.host.gc_low_watermark_pages = 0;
    config.hbf.host.gc_hard_watermark_pages = 0;
    // Automatic GC under full-resident mapping requires a one-block GC-only
    // reserve per plane (the device rejects 0); the watermarks above keep
    // GC from ever triggering in these small graphs.
    config.hbf.host.gc_reserved_free_blocks_per_plane = 1;
    config.static_hbf_blocks_per_plane = enable_hbf ? 1 : 0;
    config.base_die_link.read_bandwidth_GBps = 64.0;
    config.base_die_link.write_bandwidth_GBps = 64.0;
    config.base_die_link.latency_ns = 20.0;
    config.external.capacity_bytes = 64 * kPage;
    config.external.page_size_bytes = kPage;
    config.external.media_channels = 2;
    return config;
}

SimulationTransaction transaction(
    std::string id,
    SimulationTarget target,
    Op op,
    std::uint64_t address,
    std::uint64_t bytes,
    double issue_ns,
    std::vector<std::string> dependencies = {},
    std::uint32_t stack = 0) {
    return SimulationTransaction{
        .id = std::move(id),
        .target = target,
        .op = op,
        .addr = address,
        .bytes = bytes,
        .issue_ns = issue_ns,
        .duration_ns = 0.0,
        .dependencies = std::move(dependencies),
        .stack = stack,
    };
}

void test_long_completion_work_accumulation() {
    using hbfsim::physical::PhysicalCompletion;
    using hbfsim::physical::SimulationCompletionStats;
    static_assert(std::is_aggregate_v<SimulationCompletionStats>);
    // Valid completion timestamps, chosen to expose independent long-sum
    // roundoff. This is an observation-ledger fixture, not a DRAM scheduler.
    const PhysicalCompletion completion{
        .arrival_ns = 0.0,
        .start_ns = 1.1,
        .finish_ns = 1.1 + 93.1,
        .logical_bytes = 64,
        .physical_bytes = 64,
    };
    const std::array<double, 3> value{
        completion.start_ns - completion.arrival_ns,
        completion.finish_ns - completion.start_ns,
        completion.finish_ns - completion.arrival_ns,
    };
    // Exactly the client's existing conservation tolerances, unchanged.
    const auto close = [](double lhs, double rhs) {
        return std::abs(lhs - rhs) <=
            std::max(1e-6, 1e-10 * std::max(std::abs(lhs), std::abs(rhs)));
    };
    constexpr std::uint64_t kBatches = 50;
    constexpr std::uint64_t kPerBatch = 100000;
    constexpr std::uint64_t kCount = kBatches * kPerBatch;
    SimulationCompletionStats total;
    std::array<double, 3> naive{};
    for (std::uint64_t batch = 0; batch < kBatches; ++batch) {
        SimulationCompletionStats local;
        std::array<double, 3> naive_local{};
        for (std::uint64_t i = 0; i < kPerBatch; ++i) {
            // Use the same production operation as both native batch and
            // whole-session ledgers; no test-only compensated accumulator.
            total.record_completion(completion);
            local.record_completion(completion);
            for (std::size_t j = 0; j < value.size(); ++j) {
                naive[j] += value[j];
                naive_local[j] += value[j];
            }
        }
        require(close(naive_local[2], naive_local[0] + naive_local[1]),
                "roundoff fixture no longer has passing naive batch totals");
        require(local.transactions == kPerBatch &&
                    close(local.latency_work_ns,
                          local.queue_wait_work_ns + local.service_work_ns),
                "production batch completion work failed conservation");
    }
    require(!close(naive[2], naive[0] + naive[1]),
            "roundoff fixture did not expose old cumulative conservation failure");
    require(close(total.latency_work_ns,
                  total.queue_wait_work_ns + total.service_work_ns),
            "compensated cumulative completion work does not conserve");
    // Multiplication rounds each exact repeated binary64 input sum once.
    require(total.queue_wait_work_ns == value[0] * kCount &&
                total.service_work_ns == value[1] * kCount &&
                total.latency_work_ns == value[2] * kCount,
            "compensated completion work differs from independently known sums");
    require(total.latency_work_ns !=
                total.queue_wait_work_ns + total.service_work_ns,
            "latency was derived from queue plus service rather than observed independently");
    require(total.transactions == kCount &&
                total.logical_bytes == 64 * kCount &&
                total.physical_bytes == 64 * kCount &&
                total.min_latency_ns == value[2] && total.max_latency_ns == value[2] &&
                total.first_arrival_ns == 0.0 && total.finish_ns == completion.finish_ns,
            "stable summation changed integer counts or completion endpoints");

    SimulationCompletionStats partial;
    for (unsigned i = 0; i < 3; ++i) partial.record_completion(completion);
    require(std::any_of(partial.work_compensation_ns.begin(),
                        partial.work_compensation_ns.end(),
                        [](double correction) { return correction != 0.0; }),
            "copy fixture did not retain a nonzero compensation");
    auto copy = partial;
    for (unsigned i = 0; i < 997; ++i) {
        partial.record_completion(completion);
        copy.record_completion(completion);
    }
    require(copy.transactions == partial.transactions &&
                copy.queue_wait_work_ns == partial.queue_wait_work_ns &&
                copy.service_work_ns == partial.service_work_ns &&
                copy.latency_work_ns == partial.latency_work_ns &&
                copy.work_compensation_ns == partial.work_compensation_ns,
            "copy did not preserve completion-work compensation");
    copy = {};
    require(copy.work_compensation_ns == std::array<double, 3>{},
            "reset retained completion-work compensation");
    copy.record_completion(completion);
    require(copy.queue_wait_work_ns == value[0] &&
                copy.service_work_ns == value[1] && copy.latency_work_ns == value[2],
            "reset ledger did not restart independent work sums");
    auto invalid = completion;
    invalid.start_ns = invalid.finish_ns + 1.0;
    const auto before = copy;
    require_throws([&] { copy.record_completion(invalid); },
                   "invalid completion timestamps were accepted");
    require(copy.transactions == before.transactions &&
                copy.queue_wait_work_ns == before.queue_wait_work_ns &&
                copy.service_work_ns == before.service_work_ns &&
                copy.latency_work_ns == before.latency_work_ns &&
                copy.work_compensation_ns == before.work_compensation_ns,
            "invalid completion changed the observation ledger");
}

void test_explicit_hybrid_chain_and_barrier() {
    SimulationSession replay(base_config());
    auto hbf_read = transaction(
        "batch0/hbf-read", SimulationTarget::HbfPhysical, Op::Read,
        0, kPage, 0.0);
    auto link = transaction(
        "batch0/d2d-read", SimulationTarget::D2dHbfToHbm, Op::Read,
        0, kPage, 0.0, {hbf_read.id}, 0);
    auto install = transaction(
        "batch0/hbm-install", SimulationTarget::Hbm, Op::Write,
        4 * kPage, kPage, 0.0, {link.id});
    auto foreground = transaction(
        "batch0/hbm-user", SimulationTarget::Hbm, Op::Read,
        4 * kPage, kPage, 0.0, {install.id});
    SimulationTransaction compute{
        .id = "batch0/compute-fence",
        .target = SimulationTarget::Barrier,
        .op = Op::Read,
        .addr = 0,
        .bytes = 0,
        .issue_ns = 0.0,
        .duration_ns = 25.0,
        .dependencies = {foreground.id},
        .stack = 0,
    };
    const auto result = replay.run_batch(
        "0", {hbf_read, link, install, foreground, compute});
    require(result.transactions == 5 && result.memory_transactions == 4 &&
                result.barriers == 1 && result.dependency_edges == 4,
            "transaction graph census drifted");
    require(result.completions.size() == 4 && result.finish_ns > 25.0,
            "transaction graph did not execute its physical chain");
    require(
        result.by_target[static_cast<std::size_t>(SimulationTarget::HbfPhysical)]
                .bytes == kPage &&
            result.by_target[static_cast<std::size_t>(
                SimulationTarget::D2dHbfToHbm)].bytes == kPage &&
            result.by_target[static_cast<std::size_t>(SimulationTarget::Hbm)]
                .bytes == 2 * kPage,
        "transaction target byte accounting did not conserve");
}

void test_state_and_dependencies_persist_across_batches() {
    SimulationSession replay(base_config());
    auto write = transaction(
        "batch0/hbf-write", SimulationTarget::HbfLogical, Op::Write,
        16 * kPage, kPage, 0.0);
    const auto first = replay.run_batch("0", {write});
    auto read = transaction(
        "batch1/hbf-read", SimulationTarget::HbfLogical, Op::Read,
        16 * kPage, kPage, 0.0, {write.id});
    const auto second = replay.run_batch("1", {read});
    require(second.batch_origin_ns == first.finish_ns &&
                second.first_issue_ns == first.finish_ns &&
                second.finish_ns > first.finish_ns,
            "cross-batch completion dependency was not preserved");
    require(second.completions.front().note != "unmapped-erased-read",
            "persistent HBF state lost the prior logical write");
}

void test_nonterminal_checkpoint_is_causal_and_reusable() {
    SimulationSession replay(base_config());
    const auto first = replay.run_batch(
        "checkpoint-write-0",
        {transaction(
            "checkpoint/write-0", SimulationTarget::HbfLogical, Op::Write,
            16 * kPage, kPage, 0.0)});
    const auto checkpoint0 = replay.checkpoint_pending("periodic-0");
    require(
        checkpoint0.checkpoint_id == "periodic-0" &&
            checkpoint0.sequence == 0 &&
            checkpoint0.persistence.serving_frontier_ns == first.finish_ns &&
            checkpoint0.persistence.finish_ns > first.finish_ns &&
            checkpoint0.persistence.quiescence.quiescent() &&
            checkpoint0.persistence.device_after.hbf.mapping_page_programs >
                checkpoint0.persistence.device_before.hbf.mapping_page_programs &&
            replay.completed_frontier_ns() ==
                checkpoint0.persistence.finish_ns &&
            replay.completed_checkpoints() == 1,
        "nonterminal checkpoint did not persist and advance causally");

    const auto second = replay.run_batch(
        "checkpoint-write-1",
        {transaction(
            "checkpoint/write-1", SimulationTarget::HbfLogical, Op::Write,
            17 * kPage, kPage, 0.0)});
    require(
        second.batch_origin_ns == checkpoint0.persistence.finish_ns,
        "post-checkpoint batch did not start after the persisted frontier");
    const auto checkpoint1 = replay.checkpoint_pending("periodic-1");
    require(
        checkpoint1.sequence == 1 &&
            checkpoint1.persistence.serving_frontier_ns == second.finish_ns &&
            checkpoint1.persistence.finish_ns > second.finish_ns &&
            replay.completed_checkpoints() == 2,
        "second nonterminal checkpoint did not remain reusable");
    require_throws(
        [&] { (void)replay.checkpoint_pending("periodic-1"); },
        "duplicate checkpoint id was accepted");

    const auto drain = replay.drain_pending();
    require(
        drain.serving_frontier_ns == checkpoint1.persistence.finish_ns &&
            drain.finish_ns == drain.serving_frontier_ns &&
            drain.hbf_completion.physical_bytes == 0,
        "terminal drain repeated already checkpointed physical work");
    require_throws(
        [&] { (void)replay.checkpoint_pending("after-drain"); },
        "checkpoint was accepted after terminal drain");
}

void test_explicit_crash_observes_state_without_drain() {
    SimulationSession dirty(base_config());
    const auto write = dirty.run_batch(
        "crash-write",
        {transaction(
            "crash/write", SimulationTarget::HbfLogical, Op::Write,
            16 * kPage, kPage, 0.0)});
    const auto crash = dirty.inject_crash("after-write");
    require(
        crash.crash_id == "after-write" &&
            crash.completed_batches == 1 &&
            crash.completed_checkpoints == 0 &&
            crash.completed_frontier_ns == write.finish_ns &&
            !crash.quiescence.quiescent() &&
            (crash.quiescence.dirty_mapping_pages > 0 ||
             crash.quiescence.pending_dirty_mapping_events > 0 ||
             crash.quiescence.pending_lpn_updates > 0 ||
             crash.quiescence.pending_commits > 0 ||
             crash.quiescence.write_buffer_entries > 0 ||
             crash.quiescence.inflight_buffered_generations > 0) &&
            crash.quiescence.pending_commits == 0 &&
            crash.quiescence.pending_lpn_updates == 0 &&
            crash.device_at_injection.hbf.mapping_page_programs == 0,
        "explicit crash hid or drained uncheckpointed HBF state");
    require_throws(
        [&] { (void)dirty.drain_pending(); },
        "crashed session accepted a terminal drain");
    require_throws(
        [&] { (void)dirty.checkpoint_pending("after-crash"); },
        "crashed session accepted a checkpoint");

    SimulationSession clean(base_config());
    (void)clean.run_batch(
        "checkpointed-write",
        {transaction(
            "checkpointed/write", SimulationTarget::HbfLogical, Op::Write,
            16 * kPage, kPage, 0.0)});
    const auto checkpoint = clean.checkpoint_pending("durable");
    const auto checkpointed_crash = clean.inject_crash("after-checkpoint");
    require(
        checkpoint.persistence.quiescence.quiescent() &&
            checkpointed_crash.completed_checkpoints == 1 &&
            checkpointed_crash.completed_frontier_ns ==
                checkpoint.persistence.finish_ns &&
            checkpointed_crash.quiescence.quiescent(),
        "crash after a successful checkpoint did not retain its clean boundary");
}

void test_preloaded_logical_image_is_real_mutable_media_state() {
    auto config = base_config();
    config.static_hbf_blocks_per_plane = 1;
    config.initial_hbf_logical_first_lpn = 16;
    config.initial_hbf_logical_pages = 4;
    SimulationSession replay(std::move(config));
    const auto initial = replay.run_batch(
        "initial-read",
        {transaction(
            "initial/read", SimulationTarget::HbfLogical, Op::Read,
            16 * kPage, kPage, 0.0)});
    require(
        initial.completions.front().note != "unmapped-erased-read",
        "initial image was not installed as logical media state");
    const auto overwrite = replay.run_batch(
        "overwrite",
        {transaction(
            "initial/write", SimulationTarget::HbfLogical, Op::Write,
            16 * kPage, kPage, 0.0, {"initial/read"})});
    require(
        overwrite.finish_ns > initial.finish_ns,
        "mutable initial image did not accept an overwrite");

    auto impossible = base_config(false);
    impossible.initial_hbf_logical_pages = 1;
    require_throws(
        [&] { SimulationSession invalid(std::move(impossible)); },
        "initial HBF image was accepted with HBF disabled");
}

void test_invalid_graphs_fail_before_execution() {
    SimulationSession replay(base_config(false));
    auto first = transaction(
        "a", SimulationTarget::Hbm, Op::Read, 0, kPage, 0.0, {"b"});
    auto second = transaction(
        "b", SimulationTarget::Hbm, Op::Read, kPage, kPage, 0.0, {"a"});
    require_throws(
        [&] { (void)replay.run_batch("cycle", {first, second}); },
        "cyclic transaction graph was accepted");

    SimulationSession missing(base_config(false));
    auto orphan = transaction(
        "orphan", SimulationTarget::Hbm, Op::Read, 0, kPage, 0.0,
        {"does-not-exist"});
    require_throws(
        [&] { (void)missing.run_batch("missing", {orphan}); },
        "missing transaction dependency was accepted");

    SimulationSession semantic_free(base_config(false));
    auto outside = transaction(
        "outside", SimulationTarget::Hbm, Op::Read, 32 * kPage, kPage, 0.0);
    require_throws(
        [&] { (void)semantic_free.run_batch("outside", {outside}); },
        "out-of-capacity transaction address was accepted");
}

void test_static_extent_and_d2d_tier_guards() {
    SimulationSession static_extent(base_config());
    const auto plane_bytes = 8 * 8 * kPage;
    auto wrapped = transaction(
        "wrapped-static-read", SimulationTarget::HbfPhysical, Op::Read,
        0, plane_bytes + kPage, 0.0);
    require_throws(
        [&] { (void)static_extent.run_batch("wrapped", {wrapped}); },
        "physical HBF range escaped through a plane-boundary wrap");
    const auto valid = static_extent.run_batch(
        "valid-next-plane",
        {transaction(
            "next-plane-static-read", SimulationTarget::HbfPhysical, Op::Read,
            plane_bytes, kPage, 0.0)});
    require(valid.memory_transactions == 1,
            "declared static block on a later plane was rejected");

    auto no_hbm_config = base_config();
    no_hbm_config.enable_hbm = false;
    require_throws([&] { SimulationSession invalid(no_hbm_config); },
        "logical HBF controller accepted a topology without HBM storage");
    // Raw, unbuffered device traffic has no host mapping/storage allocation.
    no_hbm_config.hbf.host.mapping_mode = hbfsim::host::MappingMode::RawPhysical;
    no_hbm_config.hbf.host.auto_gc_enabled = false;
    no_hbm_config.hbf.host.gc_reserved_free_blocks_per_plane = 0;
    SimulationSession no_hbm(std::move(no_hbm_config));
    auto impossible_link = transaction(
        "link-without-hbm", SimulationTarget::D2dHbfToHbm, Op::Read,
        0, kPage, 0.0, {}, 0);
    require_throws(
        [&] { (void)no_hbm.run_batch("no-hbm", {impossible_link}); },
        "D2D transfer involving a disabled HBM tier was accepted");
}

void test_published_extent_program_checkpoint_and_read_only_reopen() {
    auto producer_config = base_config();
    producer_config.static_hbf_blocks_per_plane = 0;
    producer_config.published_hbf_blocks_per_plane = 1;
    SimulationSession producer(producer_config);
    const auto append = producer.run_batch(
        "published-append",
        {transaction(
            "published/page0",
            SimulationTarget::HbfPhysical,
            Op::Write,
            0,
            kPage,
            0.0)});
    require(
        append.device_after.hbf.raw_physical_programs == 1 &&
            append.device_after.hbf.raw_physical_program_payload_bytes == kPage &&
            append.device_after.hbf.mapping_update_ops == 0,
        "published append did not bypass page mapping with exact write accounting");
    const auto checkpoint = producer.checkpoint_pending("published");
    require(
        checkpoint.persistence.quiescence.quiescent(),
        "published append did not reach a durable checkpoint");
    auto image = std::make_shared<const hbfsim::host::HbfPersistentImage>(
        producer.persistent_hbf_image());
    (void)producer.inject_crash("published-boundary");

    auto recovered_config = base_config();
    recovered_config.static_hbf_blocks_per_plane = 0;
    recovered_config.published_hbf_blocks_per_plane = 1;
    recovered_config.initial_hbf_persistent_image = std::move(image);
    SimulationSession recovered(std::move(recovered_config));
    const auto direct = recovered.run_batch(
        "published-direct-read",
        {transaction(
            "published/read-page0",
            SimulationTarget::HbfPhysical,
            Op::Read,
            0,
            kPage,
            0.0)});
    require(
        direct.device_after.hbf.page_reads == 1 &&
            direct.device_after.hbf.mapping_lookup_ops == 0,
        "published reopen did not use the mapping-free physical read path");
    require_throws(
        [&] {
            (void)recovered.run_batch(
                "published-illegal-rewrite",
                {transaction(
                    "published/rewrite-page0",
                    SimulationTarget::HbfPhysical,
                    Op::Write,
                    0,
                    kPage,
                    0.0)});
        },
        "restored published extent accepted a rewrite");
}

void test_static_extent_declaration_is_restored_from_image() {
    auto producer_config = base_config();
    producer_config.static_hbf_blocks_per_plane = 1;
    SimulationSession producer(std::move(producer_config));
    const auto image = std::make_shared<const
        hbfsim::host::HbfPersistentImage>(
            producer.persistent_hbf_image());

    auto recovered_config = base_config();
    recovered_config.static_hbf_blocks_per_plane = 0;
    recovered_config.initial_hbf_persistent_image = image;
    const auto static_pages =
        recovered_config.hbf.device.stacks *
        recovered_config.hbf.device.channels_per_stack *
        recovered_config.hbf.device.dies_per_channel *
        recovered_config.hbf.device.planes_per_die *
        recovered_config.hbf.device.pages_per_block;
    SimulationSession recovered(std::move(recovered_config));
    const auto direct = recovered.run_batch(
        "restored-static-read",
        {transaction(
            "restored-static/page0",
            SimulationTarget::HbfStatic,
            Op::Read,
            0,
            kPage,
            0.0)});
    require(
        direct.device_after.hbf.page_reads == 1 &&
            direct.device_after.hbf.mapping_lookup_ops == 0,
        "restored static extent did not remain natively readable");

    require_throws(
        [&] {
            (void)recovered.run_batch(
                "restored-static-escape",
                {transaction(
                    "restored-static/outside",
                    SimulationTarget::HbfStatic,
                    Op::Read,
                    static_pages * kPage,
                    kPage,
                    0.0)});
        },
        "restored static extent accepted an out-of-range read");
}

void test_batch_relative_time_rebases_without_changing_payload_offsets() {
    SimulationSession replay(base_config(false));
    const auto first = replay.run_batch(
        "0", {transaction(
            "first", SimulationTarget::Hbm, Op::Read, 0, kPage, 100.0)});
    require(first.finish_ns > 100.0, "first batch did not advance time");
    const auto second = replay.run_batch(
        "1", {transaction(
            "second", SimulationTarget::Hbm, Op::Read,
            kPage, kPage, 0.0)});
    require(second.batch_origin_ns == first.finish_ns &&
                second.first_issue_ns == first.finish_ns,
            "batch-relative issue time was not rebased to persistent time");
}

void test_equal_ready_hbm_requests_share_the_channel_queue() {
    auto config = base_config(false);
    config.hbm.device.channels_per_stack = 1;
    config.hbm.device.pseudo_channels_per_channel = 1;
    config.hbm.timing.read_latency_ns = 0;
    config.hbm.timing.bandwidth_efficiency = 1;
    SimulationSession replay(config);
    const auto result = replay.run_batch("fair", {
        transaction("large", SimulationTarget::Hbm, Op::Read, 0, 65536, 0),
        transaction("small", SimulationTarget::Hbm, Op::Read, 65536, 64, 0)});
    double large = 0, small = 0;
    for (const auto& c : result.completions) {
        if (c.id == "large") large = c.finish_ns;
        if (c.id == "small") small = c.finish_ns;
    }
    require(small > 0 && small < large, "large HBM parent monopolized a shared channel");
}

void test_hbm_observers_do_not_change_memory_schedule() {
    auto config = base_config(false);
    config.hbm.device.channels_per_stack = 1;
    config.hbm.device.banks_per_group = 1;
    hbfsim::physical::hbm::HbmDevice encoder(config.hbm);
    const auto conflict_address = encoder.encode(
        hbfsim::physical::hbm::HbmAddress{.row = 1});
    for (const auto hit_delay : {0.0, 1.0, 100.0}) {
        const auto run = [&](bool observe) {
            SimulationSession replay(config);
            (void)replay.run_batch("warm", {
                transaction("warm", SimulationTarget::Hbm, Op::Read, 0, 64, 0.0)});
            std::vector<SimulationTransaction> requests{
                transaction("conflict", SimulationTarget::Hbm, Op::Read,
                            conflict_address, 64, 0.0),
                transaction("hit", SimulationTarget::Hbm, Op::Read, 64, 64, hit_delay),
            };
            if (observe) {
                requests.push_back(transaction(
                    "observer", SimulationTarget::Barrier, Op::Read,
                    0, 0, 0.0, {"conflict"}));
            }
            return replay.run_batch("read", requests);
        };
        const auto plain = run(false);
        const auto observed = run(true);
        require(plain.finish_ns == observed.finish_ns &&
                    plain.device_after.hbm.service_quanta == observed.device_after.hbm.service_quanta &&
                    plain.device_after.hbm.service_busy_ns == observed.device_after.hbm.service_busy_ns,
                "zero-duration HBM observer changed channel scheduling");
        require(plain.completions.size() == observed.completions.size(),
                "HBM observer changed the memory completion count");
        for (std::size_t index = 0; index < plain.completions.size(); ++index) {
            const auto& expected = plain.completions[index];
            const auto& actual = observed.completions[index];
            require(expected.id == actual.id && expected.start_ns == actual.start_ns &&
                        expected.finish_ns == actual.finish_ns,
                    "HBM observer changed a memory completion");
        }
        const auto first = std::min_element(
            plain.completions.begin(), plain.completions.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.finish_ns < rhs.finish_ns; });
        require(first->id == "conflict",
                "HBM FIFO admission reordered small equal-channel requests");
    }
}

void test_hbm_dependency_precedes_later_root_arrivals() {
    auto config = base_config(false);
    config.hbm.device.channels_per_stack = 1;
    config.hbm.device.banks_per_group = 1;
    hbfsim::physical::hbm::HbmDevice encoder(config.hbm);
    const auto conflict_address = encoder.encode(
        hbfsim::physical::hbm::HbmAddress{.row = 1});
    SimulationSession replay(config);
    const auto result = replay.run_batch("causal", {
        transaction("producer", SimulationTarget::Hbm, Op::Read, 0, 64, 0.0),
        transaction("future-root", SimulationTarget::Hbm, Op::Read,
                    conflict_address, 64, 500.0),
        transaction("consumer", SimulationTarget::Hbm, Op::Read,
                    64, 64, 0.0, {"producer"}),
    });
    require(result.completions.size() == 3 &&
                result.completions[0].id == "producer" &&
                result.completions[1].id == "consumer" &&
                result.completions[1].arrival_ns == result.completions[0].finish_ns &&
                result.completions[1].finish_ns < 500.0 &&
                result.completions[2].id == "future-root",
            "HBM deferred a ready dependency behind a later independent arrival");
}

void test_hbm_controller_time_does_not_regress_between_banks() {
    auto config = base_config(false);
    config.hbm.device.capacity_bytes = 1ull << 20;
    config.hbm.device.channels_per_stack = 1;
    config.hbm.device.bank_groups_per_pseudo_channel = 4;
    config.hbm.device.banks_per_group = 4;
    config.trace.retain_completion_diagnostics = false;
    SimulationSession replay(config);
    const auto result = replay.run_batch("bank-clock", {
        transaction("producer", SimulationTarget::Hbm, Op::Write, 68288, 192, 60.0),
        transaction("peer-0", SimulationTarget::Hbm, Op::Write, 72640, 512, 39.0),
        transaction("consumer", SimulationTarget::Hbm, Op::Read, 122880, 384, 18.0,
                    {"producer"}),
        transaction("peer-1", SimulationTarget::Hbm, Op::Write, 101504, 128, 29.0),
        transaction("peer-2", SimulationTarget::Hbm, Op::Write, 3200, 320, 47.0),
        transaction("producer-2", SimulationTarget::Hbm, Op::Read, 25728, 256, 50.0),
        transaction("peer-3", SimulationTarget::Hbm, Op::Write, 46784, 320, 56.0),
        transaction("consumer-2", SimulationTarget::Hbm, Op::Read, 119168, 512, 2.0,
                    {"producer-2"}),
    });
    require(result.completions.size() == 8,
            "HBM bank backfill lost a request or regressed a dependency timestamp");
    for (const auto& completion : result.completions) {
        require(completion.start_ns >= completion.arrival_ns &&
                    completion.finish_ns >= completion.start_ns,
                "HBM bank backfill produced a noncausal completion");
    }
}

void test_external_is_a_semantic_free_target() {
    auto config = base_config(false);
    config.enable_external = true;
    SimulationSession replay(std::move(config));
    auto fill = transaction(
        "external/fill", SimulationTarget::External, Op::Read,
        40 * kPage, 3 * kPage, 0.0);
    auto install = transaction(
        "external/install", SimulationTarget::Hbm, Op::Write,
        4 * kPage, kPage, 0.0, {fill.id});
    auto foreground = transaction(
        "external/user", SimulationTarget::Hbm, Op::Read,
        4 * kPage, kPage, 0.0, {install.id});
    const auto result = replay.run_batch(
        "external", {fill, install, foreground});
    const auto external_index = static_cast<std::size_t>(
        SimulationTarget::External);
    require(
        result.by_target[external_index].transactions == 1 &&
            result.by_target[external_index].bytes == 3 * kPage &&
            result.device_after.external.read_requests == 3 &&
            result.device_after.external.read_bytes == 3 * kPage &&
            result.device_after.external.s2m_payload_bytes == 3 * kPage &&
            result.device_after.external.page_run_requests == 1 &&
            result.device_after.external.page_run_segments == 3 &&
            result.device_after.external.page_run_pages == 3,
        "external target did not preserve device accounting");
    const auto external_completion = std::find_if(
        result.completions.begin(),
        result.completions.end(),
        [](const auto& completion) {
            return completion.id == "external/fill";
        });
    require(
        result.completions.size() == 3 &&
            external_completion != result.completions.end() &&
            external_completion->logical_bytes == 3 * kPage &&
            external_completion->physical_bytes == 3 * kPage &&
            external_completion->arrival_ns <= external_completion->start_ns &&
            external_completion->start_ns <= external_completion->finish_ns,
        "per-transaction completion lost identity, bytes, or timestamps");

    auto disabled = base_config(false);
    SimulationSession no_external(std::move(disabled));
    require_throws(
        [&] {
            (void)no_external.run_batch(
                "disabled",
                {transaction(
                    "external/disabled", SimulationTarget::External, Op::Read,
                    0, kPage, 0.0)});
        },
        "external target was accepted while disabled");
}

hbfsim::physical::BaseDieLinkConfig direct_lane_envelope() {
    return hbfsim::physical::BaseDieLinkConfig{
        .read_bandwidth_GBps = 32.0,
        .write_bandwidth_GBps = 32.0,
        .latency_ns = 100.0,
    };
}

void test_direct_hbf_external_lane_round_trip_and_guards() {
    auto config = base_config();
    config.enable_external = true;
    config.hbf_external_direct_link = direct_lane_envelope();
    SimulationSession replay(config);
    auto populate = transaction(
        "b0/populate", SimulationTarget::HbfLogical, Op::Write,
        0, kPage, 0.0);
    auto hbf_read = transaction(
        "b0/hbf-read", SimulationTarget::HbfLogical, Op::Read,
        0, kPage, 0.0, {populate.id});
    // Stack 1 exercises the per-stack lane population.
    auto lane = transaction(
        "b0/demote-lane", SimulationTarget::DirectHbfToExternal, Op::Read,
        0, kPage, 0.0, {hbf_read.id}, 1);
    auto land = transaction(
        "b0/external-write", SimulationTarget::External, Op::Write,
        0, kPage, 0.0, {lane.id});
    auto back_read = transaction(
        "b0/external-read", SimulationTarget::External, Op::Read,
        0, kPage, 0.0, {land.id});
    auto back_lane = transaction(
        "b0/restore-lane", SimulationTarget::DirectExternalToHbf, Op::Write,
        0, kPage, 0.0, {back_read.id}, 0);
    auto restore = transaction(
        "b0/restore-write", SimulationTarget::HbfLogical, Op::Write,
        kPage, kPage, 0.0, {back_lane.id});
    const auto result = replay.run_batch(
        "0",
        {populate, hbf_read, lane, land, back_read, back_lane, restore});
    require(
        result.by_target[static_cast<std::size_t>(
                SimulationTarget::DirectHbfToExternal)].bytes == kPage &&
            result.by_target[static_cast<std::size_t>(
                SimulationTarget::DirectExternalToHbf)].bytes == kPage,
        "direct lane byte census did not conserve");
    double lane_cost_ns = 0.0;
    double restore_lane_finish_ns = 0.0;
    double external_write_finish_ns = 0.0;
    for (const auto& completion : result.completions) {
        if (completion.id == "b0/demote-lane") {
            lane_cost_ns = completion.finish_ns - completion.arrival_ns;
        }
        if (completion.id == "b0/restore-lane") {
            restore_lane_finish_ns = completion.finish_ns;
        }
        if (completion.id == "b0/external-write") {
            external_write_finish_ns = completion.finish_ns;
        }
    }
    // 100 ns fixed lane latency + 4096 B / 32 GB/s = 128 ns serialization.
    require(std::abs(lane_cost_ns - 228.0) < 1e-6,
            "direct lane cost diverged from its configured envelope");
    require(restore_lane_finish_ns > external_write_finish_ns,
            "restore lane did not chain after the external landing");
    const auto snapshot = replay.device_snapshot();
    require(snapshot.hbf_external_direct_link.links == 2 &&
                snapshot.hbf_external_direct_link.read_bytes == kPage &&
                snapshot.hbf_external_direct_link.write_bytes == kPage,
            "direct lane device accounting drifted");

    const auto reject = [](SimulationSessionConfig config_variant,
                           SimulationTransaction bad) {
        SimulationSession guard(std::move(config_variant));
        try {
            (void)guard.run_batch("guard", {std::move(bad)});
        } catch (const hbfsim::physical::SimulationInputError&) {
            return true;
        }
        return false;
    };
    auto external_config = base_config();
    external_config.enable_external = true;
    external_config.hbf_external_direct_link = direct_lane_envelope();
    require(
        reject(
            external_config,
            transaction(
                "g0/wrong-op", SimulationTarget::DirectHbfToExternal,
                Op::Write, 0, kPage, 0.0, {}, 0)),
        "direct offload lane accepted the restore direction");
    require(
        reject(
            external_config,
            transaction(
                "g1/stack", SimulationTarget::DirectExternalToHbf,
                Op::Write, 0, kPage, 0.0, {}, 7)),
        "direct lane accepted an out-of-range stack");
    auto lane_less_config = base_config();
    lane_less_config.enable_external = true;
    require(
        reject(
            lane_less_config,
            transaction(
                "g2/no-lane", SimulationTarget::DirectHbfToExternal,
                Op::Read, 0, kPage, 0.0, {}, 0)),
        "direct lane existed without an explicit envelope");
    require(
        replay.device_snapshot().hbf_external_direct_link.links == 2 &&
            SimulationSession(lane_less_config)
                    .device_snapshot().hbf_external_direct_link.links == 0,
        "direct lane instantiation did not follow the explicit envelope");
    auto lane_config_without_external = base_config();
    lane_config_without_external.hbf_external_direct_link =
        direct_lane_envelope();
    require(
        reject(
            lane_config_without_external,
            transaction(
                "g3/no-external", SimulationTarget::DirectHbfToExternal,
                Op::Read, 0, kPage, 0.0, {}, 0)),
        "direct lane accepted a session without the external tier");
}

void test_frontier_excludes_detached_work_from_the_next_origin() {
    using hbfsim::physical::SimulationBatchOptions;
    using hbfsim::physical::SimulationInputError;
    auto config = base_config();
    config.enable_external = true;
    SimulationSession replay(config);
    const auto finish_of = [](const auto& result, const std::string& id) {
        const auto found = std::find_if(
            result.completions.begin(), result.completions.end(),
            [&](const auto& completion) { return completion.id == id; });
        require(found != result.completions.end(), "completion missing: " + id);
        return *found;
    };

    // Blocking HBM read plus a terminal external write nobody in the batch
    // waits for. Only the read defines the next origin.
    auto blocking = transaction(
        "s1/read", SimulationTarget::Hbm, Op::Read, 0, kPage, 0.0);
    auto offload = transaction(
        "s1/offload", SimulationTarget::External, Op::Write,
        0, 16 * kPage, 0.0);
    const auto first = replay.run_batch(
        "s1", {blocking, offload},
        SimulationBatchOptions{.frontier = std::vector<std::string>{"s1/read"}});
    const auto read_done = finish_of(first, "s1/read").finish_ns;
    const auto offload_done = finish_of(first, "s1/offload").finish_ns;
    require(
        first.frontier_transactions == 1 &&
            first.blocking_finish_ns == read_done &&
            first.finish_ns == offload_done &&
            offload_done > read_done &&
            replay.completed_frontier_ns() == read_done,
        "frontier did not separate blocking from detached completion");

    // A dependency-free batch starts at the blocking frontier ...
    const auto second = replay.run_batch(
        "s2",
        {transaction(
            "s2/read", SimulationTarget::Hbm, Op::Read, kPage, kPage, 0.0)});
    require(
        second.batch_origin_ns == read_done &&
            finish_of(second, "s2/read").arrival_ns == read_done,
        "next batch was delayed by detached work");
    // ... while a later dependent of the detached write still waits for it.
    const auto third = replay.run_batch(
        "s3",
        {transaction(
            "s3/restore", SimulationTarget::External, Op::Read,
            0, kPage, 0.0, {"s1/offload"})});
    require(
        finish_of(third, "s3/restore").arrival_ns >= offload_done,
        "cross-batch dependency on detached work was lost");

    // The frontier is never earlier than the batch's last arrival: a detached
    // chain that becomes ready late still moves the next origin past it, so
    // every device keeps seeing nondecreasing arrivals.
    SimulationTransaction delay{
        .id = "s4/delay",
        .target = SimulationTarget::Barrier,
        .op = Op::Read,
        .addr = 0,
        .bytes = 0,
        .issue_ns = 0.0,
        .duration_ns = 5000.0,
        .dependencies = {},
        .stack = 0,
    };
    auto late_offload = transaction(
        "s4/late-offload", SimulationTarget::External, Op::Write,
        20 * kPage, kPage, 0.0, {"s4/delay"});
    auto early_read = transaction(
        "s4/read", SimulationTarget::Hbm, Op::Read, 2 * kPage, kPage, 0.0);
    const auto fourth = replay.run_batch(
        "s4", {delay, late_offload, early_read},
        SimulationBatchOptions{.frontier = std::vector<std::string>{"s4/read"}});
    require(
        fourth.blocking_finish_ns >= fourth.batch_origin_ns + 5000.0 &&
            fourth.blocking_finish_ns <
                finish_of(fourth, "s4/late-offload").finish_ns,
        "frontier ignored a detached arrival after the frontier completion");

    // An empty frontier blocks on nothing beyond the last arrival.
    const auto fifth = replay.run_batch(
        "s5",
        {transaction(
            "s5/offload", SimulationTarget::External, Op::Write,
            24 * kPage, kPage, 0.0)},
        SimulationBatchOptions{.frontier = std::vector<std::string>{}});
    require(
        fifth.frontier_transactions == 0 &&
            fifth.blocking_finish_ns == fifth.batch_origin_ns &&
            fifth.finish_ns > fifth.blocking_finish_ns,
        "empty frontier did not advance only past the last arrival");

    // Persistence covers all issued work, detached or not.
    const auto checkpoint = replay.checkpoint_pending("after-detached");
    require(
        checkpoint.persistence.serving_frontier_ns == fifth.finish_ns,
        "checkpoint started before detached work landed");

    require_throws(
        [&] {
            (void)replay.run_batch(
                "s6",
                {transaction(
                    "s6/read", SimulationTarget::Hbm, Op::Read,
                    0, kPage, 0.0)},
                SimulationBatchOptions{
                    .frontier = std::vector<std::string>{"s6/absent"}});
        },
        "frontier accepted an id outside the batch");
    require_throws(
        [&] {
            (void)replay.run_batch(
                "s6",
                {transaction(
                    "s6/read", SimulationTarget::Hbm, Op::Read,
                    0, kPage, 0.0)},
                SimulationBatchOptions{
                    .frontier = std::vector<std::string>{"s6/read", "s6/read"}});
        },
        "frontier accepted a repeated id");

    // Completions can be left out of the result without losing accounting.
    const auto compact = replay.run_batch(
        "s7",
        {transaction(
            "s7/read", SimulationTarget::Hbm, Op::Read, 0, kPage, 0.0)},
        SimulationBatchOptions{.record_completions = false});
    require(
        compact.completions.empty() &&
            compact.completion_by_target[static_cast<std::size_t>(
                SimulationTarget::Hbm)][0].transactions == 1 &&
            compact.finish_ns > compact.batch_origin_ns,
        "completion opt-out dropped aggregate accounting");
}

void test_input_errors_leave_the_session_usable() {
    using hbfsim::physical::SimulationInputError;
    SimulationSession replay(base_config(false));
    const auto expect_input_error = [&](auto&& function, const std::string& what) {
        try {
            function();
        } catch (const SimulationInputError&) {
            return;
        } catch (const std::runtime_error&) {
            throw std::runtime_error(what + " was not a SimulationInputError");
        }
        throw std::runtime_error(what + " was accepted");
    };
    expect_input_error(
        [&] { (void)replay.run_batch("empty", {}); },
        "empty batch");
    expect_input_error(
        [&] {
            (void)replay.run_batch(
                "bad-address",
                {transaction(
                    "bad/read", SimulationTarget::Hbm, Op::Read,
                    64 * kPage, kPage, 0.0)});
        },
        "out-of-capacity address");
    expect_input_error(
        [&] {
            (void)replay.run_batch(
                "bad-dependency",
                {transaction(
                    "bad/read", SimulationTarget::Hbm, Op::Read,
                    0, kPage, 0.0, {"nowhere"})});
        },
        "unknown dependency");
    expect_input_error(
        [&] {
            (void)replay.run_batch(
                "bad-tier",
                {transaction(
                    "bad/hbf", SimulationTarget::HbfLogical, Op::Read,
                    0, kPage, 0.0)});
        },
        "disabled tier");
    require(
        replay.completed_batches() == 0 &&
            replay.device_snapshot().hbm.read_bytes == 0 &&
            replay.completed_frontier_ns() == 0.0,
        "rejected batches touched session state");
    // A logical HBF range beyond the device's logical capacity is a caller
    // input error as well: rejected before any device state changes, and
    // the session keeps accepting work inside the capacity.
    {
        SimulationSession with_hbf(base_config(true));
        const auto capacity_pages =
            with_hbf.device_snapshot().hbf.logical_capacity_pages;
        constexpr std::uint64_t raw_pages = 2 * 1 * 1 * 2 * 8 * 8;
        require(capacity_pages > 0 && capacity_pages < raw_pages,
                "test geometry must derive a logical capacity below its raw pages");
        expect_input_error(
            [&] {
                (void)with_hbf.run_batch(
                    "beyond-capacity",
                    {transaction(
                        "bad/hbf-logical", SimulationTarget::HbfLogical,
                        Op::Write, capacity_pages * kPage, kPage, 0.0)});
            },
            "HBF logical range beyond the logical capacity");
        require(
            with_hbf.completed_batches() == 0 &&
                with_hbf.device_snapshot().hbf.logical_write_bytes == 0,
            "capacity rejection touched HBF state");
        const auto inside = with_hbf.run_batch(
            "inside-capacity",
            {transaction(
                "good/hbf-logical", SimulationTarget::HbfLogical, Op::Write,
                0, kPage, 0.0)});
        require(inside.sequence == 0 && with_hbf.completed_batches() == 1,
                "session was not usable after the capacity rejection");
    }
    // The rejected batch ids and transaction ids were never consumed.
    const auto result = replay.run_batch(
        "bad-address",
        {transaction(
            "bad/read", SimulationTarget::Hbm, Op::Read, 0, kPage, 0.0)});
    require(
        result.sequence == 0 && replay.completed_batches() == 1,
        "session did not continue after rejected input");
}

void check_zero_capacity_rejection(const SimulationSessionConfig& config) {
    SimulationSession session(config);
    require(session.device_snapshot().hbf.logical_capacity_pages == 0,
            "fixture must expose zero logical capacity");
    bool rejected = false;
    try {
        (void)session.run_batch("bad", {
            transaction("first", SimulationTarget::Hbm, Op::Read, 0, kPage, 0.0),
            transaction("invalid", SimulationTarget::HbfLogical, Op::Write, 0, kPage, 10000.0),
        });
    } catch (const hbfsim::physical::SimulationInputError&) {
        rejected = true;
    }
    require(rejected && session.device_snapshot().hbm.read_bytes == 0 &&
                session.completed_batches() == 0,
            "zero-capacity input executed a partial batch");
    const auto valid = session.run_batch("bad", {
        transaction("first", SimulationTarget::Hbm, Op::Read, 0, kPage, 0.0),
    });
    require(valid.sequence == 0, "zero-capacity rejection poisoned the session");
}

void test_zero_logical_capacity_rejects_the_whole_batch() {
    for (int reservation = 0; reservation < 3; ++reservation) {
        auto config = base_config(true);
        config.hbf.device.stacks = 1;
        config.hbf.device.planes_per_die = 1;
        config.hbf.device.blocks_per_plane = reservation == 0 ? 5 : 8;
        config.hbf.device.pages_per_block = 4;
        config.hbf.host.gc_reserved_free_blocks_per_plane = 2;
        config.static_hbf_blocks_per_plane = reservation == 1 ? 3 : 0;
        config.published_hbf_blocks_per_plane = reservation == 2 ? 3 : 0;
        check_zero_capacity_rejection(config);
    }
}

void test_dependency_window_and_retain() {
    using hbfsim::physical::SimulationBatchOptions;
    using hbfsim::physical::SimulationInputError;
    using hbfsim::physical::kDependencyWindowBatches;
    SimulationSession replay(base_config(false));
    const auto read = [](const std::string& id,
                         std::vector<std::string> dependencies = {}) {
        return transaction(
            id, SimulationTarget::Hbm, Op::Read, 0, kPage, 0.0,
            std::move(dependencies));
    };
    const auto rejected = [&](const std::string& batch_id,
                              const SimulationTransaction& bad,
                              SimulationBatchOptions options = {}) {
        try {
            (void)replay.run_batch(batch_id, {bad}, options);
        } catch (const SimulationInputError&) {
            return true;
        }
        return false;
    };

    (void)replay.run_batch("w0", {read("a")});
    (void)replay.run_batch("w1", {read("b")});
    require(replay.resolvable_dependency_ids() == 2,
            "window did not hold two batches");
    (void)replay.run_batch("w2", {read("c")});
    require(replay.resolvable_dependency_ids() == kDependencyWindowBatches,
            "window grew beyond the documented batch count");
    require(
        rejected("w3", read("d", {"a"})),
        "dependency older than the window was accepted");
    const auto within = replay.run_batch("w3", {read("d", {"b"})});
    require(within.dependency_edges == 1,
            "dependency inside the window was rejected");

    // Retain keeps a named id resolvable for as long as it stays declared.
    (void)replay.run_batch(
        "r0", {read("keep")},
        SimulationBatchOptions{.retain = std::vector<std::string>{"keep"}});
    (void)replay.run_batch(
        "r1", {read("e")},
        SimulationBatchOptions{.retain = std::vector<std::string>{"keep"}});
    (void)replay.run_batch(
        "r2", {read("f")},
        SimulationBatchOptions{.retain = std::vector<std::string>{"keep"}});
    (void)replay.run_batch(
        "r3", {read("g", {"keep"})},
        SimulationBatchOptions{.retain = std::vector<std::string>{"keep"}});
    require(replay.resolvable_dependency_ids() == kDependencyWindowBatches + 1,
            "retained set did not stay bounded by the declaration");
    require(
        rejected("r4", read("h"),
                 SimulationBatchOptions{
                     .retain = std::vector<std::string>{"absent"}}),
        "retain accepted an unresolvable id");
    // Batches that carry no retain declaration (another producer sharing
    // the session, or a caller that holds nothing new) leave the retained
    // set as it is; only an explicit list replaces it.
    (void)replay.run_batch("r4", {read("h")});
    (void)replay.run_batch("r5", {read("i")});
    const auto still_kept = replay.run_batch("r6", {read("j", {"keep"})});
    require(still_kept.dependency_edges == 1,
            "retained id was forgotten by a batch without a retain list");
    require(replay.resolvable_dependency_ids() == kDependencyWindowBatches + 1,
            "retained set changed without a declaration");
    (void)replay.run_batch(
        "r7", {read("k")},
        SimulationBatchOptions{.retain = std::vector<std::string>{}});
    require(replay.resolvable_dependency_ids() == kDependencyWindowBatches,
            "an empty retain list did not clear the retained set");
    (void)replay.run_batch("r8", {read("l")});
    require(
        rejected("r9", read("m", {"keep"})),
        "id stayed resolvable after the retain list was cleared");
    // An id that left every table may be reused; uniqueness holds only
    // among resolvable ids.
    (void)replay.run_batch("reuse", {read("a")});
    require(
        rejected("dup", read("a")),
        "id repeated while still resolvable");
}

void test_equal_time_mutable_hbf_precedes_static_page_run() {
    auto config = base_config();
    config.hbf.device.blocks_per_plane = 32;
    config.static_hbf_blocks_per_plane = 16;



    config.trace.retain_completion_diagnostics = false;
    SimulationSession replay(std::move(config));
    const auto result = replay.run_batch(
        "priority",
        {
            transaction(
                "static-first-in-payload",
                SimulationTarget::HbfPhysical,
                Op::Read,
                0,
                128 * kPage,
                0.0),
            transaction(
                "mutable-second-in-payload",
                SimulationTarget::HbfLogical,
                Op::Write,
                0,
                kPage,
                0.0),
        });
    require(
        result.completions.size() == 2,
        "equal-time HBF scheduling lost a completion");
    require(
        result.completions.front().id == "mutable-second-in-payload",
        "equal-time HBF scheduling did not prioritize mutable traffic");
    require(
        result.hbf_read_engine.scalar_read_requests == 1 &&
            result.hbf_read_engine.scalar_read_pages == 128,
        "equal-time HBF scheduling did not audit all static pages");
}

} // namespace

int main() {
    try {
        test_long_completion_work_accumulation();
        test_explicit_hybrid_chain_and_barrier();
        test_state_and_dependencies_persist_across_batches();
        test_nonterminal_checkpoint_is_causal_and_reusable();
        test_explicit_crash_observes_state_without_drain();
        test_preloaded_logical_image_is_real_mutable_media_state();
        test_invalid_graphs_fail_before_execution();
        test_static_extent_and_d2d_tier_guards();
        test_published_extent_program_checkpoint_and_read_only_reopen();
        test_static_extent_declaration_is_restored_from_image();
        test_batch_relative_time_rebases_without_changing_payload_offsets();
        test_equal_ready_hbm_requests_share_the_channel_queue();
        test_hbm_observers_do_not_change_memory_schedule();
        test_hbm_dependency_precedes_later_root_arrivals();
        test_hbm_controller_time_does_not_regress_between_banks();
        test_external_is_a_semantic_free_target();
        test_direct_hbf_external_lane_round_trip_and_guards();
        test_equal_time_mutable_hbf_precedes_static_page_run();
        test_frontier_excludes_detached_work_from_the_next_origin();
        test_input_errors_leave_the_session_usable();
        test_zero_logical_capacity_rejects_the_whole_batch();
        test_dependency_window_and_retain();
        std::cout << "simulation session regression passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "simulation session regression failed: "
                  << error.what() << '\n';
        return 1;
    }
}
