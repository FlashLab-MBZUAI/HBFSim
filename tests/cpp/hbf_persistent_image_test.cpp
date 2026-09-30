#include "../../verification/probes/hbf_with_hbm.hpp"
#include "host/hbf_controller.hpp"
#include "host/hbf_persistent_image.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using hbfsim::physical::AddressSpace;
using hbfsim::physical::Op;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;
using hbfsim::host::HbfAuditSnapshot;
using hbfsim::host::HbfConfig;
// Media images intentionally omit volatile HBM striping and clock state.
// Use one buffer channel and a clock-aligned restart boundary so exact
// continuation checks isolate the persisted FTL state.
class HbfController : public hbfsim::verification::HbfWithHbm {
    static hbfsim::physical::hbm::HbmConfig memory_config() {
        hbfsim::physical::hbm::HbmConfig config;
        config.device.stacks = 1;
        config.device.channels_per_stack = 1;
        config.device.pseudo_channels_per_channel = 1;
        return config;
    }
public:
    explicit HbfController(HbfConfig config)
        : HbfWithHbm(std::move(config), nullptr, memory_config()) {}
};
using hbfsim::host::HbfPersistentImage;
using hbfsim::host::HbfStats;
using hbfsim::host::read_persistent_image_file;
using hbfsim::host::write_persistent_image_file;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_equal(
    std::uint64_t actual,
    std::uint64_t expected,
    const std::string& context) {
    require(
        actual == expected,
        context + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
}

void require_close(double actual, double expected, const std::string& context) {
    require(
        std::abs(actual - expected) <=
            1e-12 * std::max({1.0, std::abs(actual), std::abs(expected)}),
        context + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
}

template <typename Callback>
void require_throws(Callback&& callback, const std::string& fragment) {
    try {
        callback();
    } catch (const std::runtime_error& error) {
        require(
            std::string(error.what()).find(fragment) != std::string::npos,
            "unexpected failure: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected failure containing: " + fragment);
}

HbfConfig persistence_config() {
    HbfConfig config;
    config.device.channels_per_stack = 1;
    config.device.dies_per_channel = 1;
    config.device.planes_per_die = 1;
    config.device.blocks_per_plane = 8;
    config.device.pages_per_block = 4;
    config.device.page_size_bytes = 4096;
    config.device.oob_bytes_per_page = 0;
    config.host.mapping_entries_per_page = 4;

    config.host.gc_low_watermark_pages = 8;
    config.device.ecc_decode_latency_ns = 1.0;
    config.device.ecc_encode_latency_ns = 1.0;
    config.device.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    config.device.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    return config;
}

PhysicalRequest write_request(
    std::string id,
    double arrival_ns,
    std::uint64_t lpn,
    std::uint64_t page_size_bytes) {
    return PhysicalRequest{
        .id = std::move(id),
        .tier = Tier::HBF,
        .op = Op::Write,
        .address_space = AddressSpace::Logical,
        .arrival_ns = arrival_ns,
        .addr = lpn * page_size_bytes,
        .bytes = page_size_bytes,
    };
}

void compare_audit(
    const HbfAuditSnapshot& actual,
    const HbfAuditSnapshot& expected) {
    require_equal(actual.free_pages, expected.free_pages, "free pages");
    require(
        actual.free_pages_per_stack == expected.free_pages_per_stack,
        "per-stack free pages diverged");
    require(
        actual.data_allocation_cursors == expected.data_allocation_cursors &&
            actual.mapping_allocation_cursors ==
                expected.mapping_allocation_cursors &&
            actual.gc_allocation_cursors == expected.gc_allocation_cursors,
        "allocation cursors diverged");
    require_equal(
        actual.logical_mappings.size(),
        expected.logical_mappings.size(),
        "logical mapping count");
    for (std::size_t index = 0;
         index < expected.logical_mappings.size();
         ++index) {
        require(
            actual.logical_mappings[index].key ==
                    expected.logical_mappings[index].key &&
                actual.logical_mappings[index].ppn ==
                    expected.logical_mappings[index].ppn,
            "logical mapping diverged");
    }
    require_equal(
        actual.mapping_pages.size(),
        expected.mapping_pages.size(),
        "mapping-page count");
    for (std::size_t index = 0;
         index < expected.mapping_pages.size();
         ++index) {
        require(
            actual.mapping_pages[index].key == expected.mapping_pages[index].key &&
                actual.mapping_pages[index].ppn == expected.mapping_pages[index].ppn,
            "mapping-page directory diverged");
    }
    require_equal(
        actual.materialized_pages.size(),
        expected.materialized_pages.size(),
        "materialized page count");
    for (std::size_t index = 0;
         index < expected.materialized_pages.size();
         ++index) {
        const auto& lhs = actual.materialized_pages[index];
        const auto& rhs = expected.materialized_pages[index];
        require(
            lhs.ppn == rhs.ppn && lhs.status == rhs.status &&
                lhs.owner == rhs.owner &&
                lhs.logical_key == rhs.logical_key &&
                lhs.block_epoch == rhs.block_epoch,
            "materialized page state diverged");
    }
    require_equal(actual.blocks.size(), expected.blocks.size(), "block count");
    for (std::size_t index = 0; index < expected.blocks.size(); ++index) {
        const auto& lhs = actual.blocks[index];
        const auto& rhs = expected.blocks[index];
        require(
            lhs.block == rhs.block && lhs.role == rhs.role &&
                lhs.valid_pages == rhs.valid_pages &&
                lhs.invalid_pages == rhs.invalid_pages &&
                lhs.free_pages == rhs.free_pages &&
                lhs.next_page == rhs.next_page &&
                lhs.erase_count == rhs.erase_count &&
                lhs.pending_program_pages == rhs.pending_program_pages &&
                lhs.pending_mapping_publications ==
                    rhs.pending_mapping_publications &&
                lhs.epoch == rhs.epoch &&
                lhs.erase_pending == rhs.erase_pending,
            "block state diverged");
    }
    require(actual.quiescent() && expected.quiescent(), "image is not quiescent");
}

void compare_image(
    const HbfPersistentImage& actual,
    const HbfPersistentImage& expected) {
    require(
        actual.version == expected.version &&
            actual.stacks == expected.stacks &&
            actual.planes == expected.planes &&
            actual.blocks_per_plane == expected.blocks_per_plane &&
            actual.pages_per_block == expected.pages_per_block &&
            actual.page_size_bytes == expected.page_size_bytes &&
            actual.mapping_entries_per_page ==
                expected.mapping_entries_per_page,
        "persistent geometry diverged");
    compare_audit(actual.state, expected.state);
    require_equal(
        actual.plane_state.size(), expected.plane_state.size(), "plane count");
    for (std::size_t index = 0; index < expected.plane_state.size(); ++index) {
        const auto& lhs = actual.plane_state[index];
        const auto& rhs = expected.plane_state[index];
        require(
            lhs.free_blocks == rhs.free_blocks &&
                lhs.active_data_block == rhs.active_data_block &&
                lhs.active_mapping_block == rhs.active_mapping_block &&
                lhs.active_gc_block == rhs.active_gc_block,
            "persistent plane state diverged");
    }
    require(
        actual.compact_image == expected.compact_image,
        "persistent compact directory diverged");
}

struct SeededDevice {
    HbfController device;
    double frontier_ns = 0.0;
};

SeededDevice build_seeded_device(const HbfConfig& config) {
    SeededDevice seeded{.device = HbfController(config)};
    seeded.device.prepopulate_logical_pages({0, 1, 2, 3});
    for (std::uint64_t index = 0; index < 16; ++index) {
        const auto completion = seeded.device.issue(write_request(
            "seed/write/" + std::to_string(index), seeded.frontier_ns,
            index % 5, config.device.page_size_bytes));
        seeded.frontier_ns = seeded.device.drain_pending(
            "seed/checkpoint/" + std::to_string(index),
            completion.finish_ns).finish_ns;
    }
    require(seeded.device.stats().gc_runs > 0, "seed did not exercise GC");
    require(seeded.device.stats().block_erase_count_sum > 0, "seed wear is empty");
    const auto& memory = seeded.device.buffer_memory().config();
    seeded.frontier_ns = memory.command_clock_time_ns(
        memory.command_clock_cycles(seeded.frontier_ns));
    return seeded;
}

SeededDevice build_compact_seeded_device(const HbfConfig& config) {
    SeededDevice seeded{.device = HbfController(config)};
    seeded.device.prepopulate_mutable_logical_page_range(0, 4);
    const auto stats = seeded.device.stats();
    require_equal(
        stats.compact_live_logical_data_pages,
        4,
        "compact live data before recovery");
    require_equal(
        stats.compact_retired_logical_data_pages,
        0,
        "compact retired data before recovery");
    require_equal(
        stats.compact_retired_mapping_pages,
        0,
        "compact retired mapping before recovery");
    return seeded;
}

void compare_delta(
    const HbfStats& before_a,
    const HbfStats& after_a,
    const HbfStats& before_b,
    const HbfStats& after_b) {
#define REQUIRE_SAME_DELTA(field) \
    require_equal( \
        after_a.field - before_a.field, \
        after_b.field - before_b.field, \
        "delta." #field)
    REQUIRE_SAME_DELTA(logical_write_bytes);
    REQUIRE_SAME_DELTA(physical_read_bytes);
    REQUIRE_SAME_DELTA(physical_write_bytes);
    REQUIRE_SAME_DELTA(data_program_payload_bytes);
    REQUIRE_SAME_DELTA(mapping_program_payload_bytes);
    REQUIRE_SAME_DELTA(gc_relocation_payload_bytes);
    REQUIRE_SAME_DELTA(mapping_lookup_ops);
    REQUIRE_SAME_DELTA(mapping_update_ops);
    REQUIRE_SAME_DELTA(gc_runs);
    REQUIRE_SAME_DELTA(gc_relocations);
    REQUIRE_SAME_DELTA(block_erases);
#undef REQUIRE_SAME_DELTA
}

void test_quiescent_image_restores_exact_continuation() {
    const auto config = persistence_config();
    auto uninterrupted = build_seeded_device(config);
    const auto image = uninterrupted.device.persistent_image();
    require(uninterrupted.device.stats().invalid_pages > 0,
            "recovery test must include implicit invalid occupancy");
    for (const auto& page : image.state.materialized_pages) {
        require(page.status != "invalid", "image retained dead page identities");
    }

    HbfController recovered(config);
    recovered.restore_persistent_image(image);
    compare_image(recovered.persistent_image(), image);
    require_equal(
        recovered.stats().block_erase_count_sum,
        uninterrupted.device.stats().block_erase_count_sum,
        "restored wear baseline");
    require_equal(recovered.stats().block_erases, 0, "recovery-window erases");

    const auto before_uninterrupted = uninterrupted.device.stats();
    const auto before_recovered = recovered.stats();
    const auto continued = uninterrupted.device.issue(write_request(
        "continue/overwrite2",
        uninterrupted.frontier_ns,
        2,
        config.device.page_size_bytes));
    const auto resumed = recovered.issue(write_request(
        "continue/overwrite2", 0.0, 2, config.device.page_size_bytes));
    require_close(
        continued.start_ns - uninterrupted.frontier_ns,
        resumed.start_ns,
        "continued write start");
    require_close(
        continued.finish_ns - uninterrupted.frontier_ns,
        resumed.finish_ns,
        "continued write finish");
    require_equal(
        continued.physical_bytes, resumed.physical_bytes, "continued write bytes");

    const auto continued_checkpoint = uninterrupted.device.drain_pending(
        "continue/checkpoint", continued.finish_ns);
    const auto resumed_checkpoint = recovered.drain_pending(
        "continue/checkpoint", resumed.finish_ns);
    require_close(
        continued_checkpoint.finish_ns - uninterrupted.frontier_ns,
        resumed_checkpoint.finish_ns,
        "continued checkpoint finish");
    require_equal(
        continued_checkpoint.physical_bytes,
        resumed_checkpoint.physical_bytes,
        "continued checkpoint bytes");
    compare_delta(
        before_uninterrupted,
        uninterrupted.device.stats(),
        before_recovered,
        recovered.stats());
    compare_image(
        recovered.persistent_image(),
        uninterrupted.device.persistent_image());
}

void test_multichannel_restart_preserves_media_with_cold_hbm() {
    const auto config = persistence_config();
    hbfsim::verification::HbfWithHbm warm(config);
    warm.prepopulate_logical_pages({0, 1, 2, 3});
    auto ready = warm.issue(write_request("seed", 0.0, 0, 512)).finish_ns;
    ready = warm.drain_pending("seed/drain", ready).finish_ns;
    hbfsim::verification::HbfWithHbm cold(config);
    cold.restore_persistent_image(warm.persistent_image());
    require(warm.buffer_memory().stats().controller_buffer_read_bytes > 0,
            "restart fixture did not warm the shared HBM path");
    require_equal(cold.buffer_memory().stats().controller_buffer_read_bytes, 0,
                  "restored image must not restore volatile HBM traffic");
    const auto before_warm = warm.stats();
    const auto before_cold = cold.stats();
    double cold_ready = 0.0;
    for (std::uint64_t lpn = 0; lpn < 4; ++lpn) {
        const auto id = "continue/" + std::to_string(lpn);
        ready = warm.issue(write_request(id, ready, lpn, 512)).finish_ns;
        cold_ready = cold.issue(write_request(id, cold_ready, lpn, 512)).finish_ns;
        ready = warm.drain_pending(id + "/drain", ready).finish_ns;
        cold_ready = cold.drain_pending(id + "/drain", cold_ready).finish_ns;
    }
    // HBM clock phase and striping are volatile. Compare durable state and
    // media work without requiring a cold controller to reproduce warm timing.
    compare_delta(before_warm, warm.stats(), before_cold, cold.stats());
    compare_image(cold.persistent_image(), warm.persistent_image());
}

void test_compact_image_restores_exact_gc_continuation() {
    const auto config = persistence_config();
    auto uninterrupted = build_compact_seeded_device(config);
    const auto image = uninterrupted.device.persistent_image();
    require(image.version == 7, "persistent image did not use v7");
    require(image.compact_image.has_value(), "compact directory is missing");

    HbfController recovered(config);
    recovered.restore_persistent_image(image);
    compare_image(recovered.persistent_image(), image);
    require_equal(
        recovered.stats().compact_live_logical_data_pages,
        4,
        "restored compact live data");

    const auto before_uninterrupted = uninterrupted.device.stats();
    const auto before_recovered = recovered.stats();
    const auto continued_overwrite = uninterrupted.device.issue(write_request(
        "compact/overwrite0",
        uninterrupted.frontier_ns,
        0,
        config.device.page_size_bytes));
    const auto resumed_overwrite = recovered.issue(write_request(
        "compact/overwrite0", 0.0, 0, config.device.page_size_bytes));
    require_close(
        continued_overwrite.finish_ns - uninterrupted.frontier_ns,
        resumed_overwrite.finish_ns,
        "compact overwrite finish");
    require_equal(
        continued_overwrite.physical_bytes,
        resumed_overwrite.physical_bytes,
        "compact overwrite bytes");
    const auto continued_first_checkpoint =
        uninterrupted.device.drain_pending(
            "compact/checkpoint0", continued_overwrite.finish_ns);
    const auto resumed_first_checkpoint = recovered.drain_pending(
        "compact/checkpoint0", resumed_overwrite.finish_ns);
    require_close(
        continued_first_checkpoint.finish_ns - uninterrupted.frontier_ns,
        resumed_first_checkpoint.finish_ns,
        "compact first checkpoint finish");

    const auto continued = uninterrupted.device.issue(write_request(
        "compact/append4",
        continued_first_checkpoint.finish_ns,
        4,
        config.device.page_size_bytes));
    const auto resumed = recovered.issue(write_request(
        "compact/append4",
        resumed_first_checkpoint.finish_ns,
        4,
        config.device.page_size_bytes));
    require_close(
        continued.finish_ns - continued_first_checkpoint.finish_ns,
        resumed.finish_ns - resumed_first_checkpoint.finish_ns,
        "compact continued write finish");
    require_equal(
        continued.physical_bytes,
        resumed.physical_bytes,
        "compact continued write bytes");

    const auto continued_checkpoint = uninterrupted.device.drain_pending(
        "compact/checkpoint1", continued.finish_ns);
    const auto resumed_checkpoint = recovered.drain_pending(
        "compact/checkpoint1", resumed.finish_ns);
    require_close(
        continued_checkpoint.finish_ns -
            continued_first_checkpoint.finish_ns,
        resumed_checkpoint.finish_ns - resumed_first_checkpoint.finish_ns,
        "compact continued checkpoint finish");
    require_equal(
        continued_checkpoint.physical_bytes,
        resumed_checkpoint.physical_bytes,
        "compact continued checkpoint bytes");
    double continued_frontier = continued_checkpoint.finish_ns;
    double resumed_frontier = resumed_checkpoint.finish_ns;
    for (std::uint64_t index = 0; index < 32; ++index) {
        const auto identifier = "compact/gc/" + std::to_string(index);
        const auto continued_write = uninterrupted.device.issue(write_request(
            identifier, continued_frontier, index % 5, config.device.page_size_bytes));
        const auto resumed_write = recovered.issue(write_request(
            identifier, resumed_frontier, index % 5, config.device.page_size_bytes));
        require_close(
            continued_write.finish_ns - continued_frontier,
            resumed_write.finish_ns - resumed_frontier,
            "compact GC continuation latency");
        const auto continued_drain = uninterrupted.device.drain_pending(
            identifier + "/checkpoint", continued_write.finish_ns);
        const auto resumed_drain = recovered.drain_pending(
            identifier + "/checkpoint", resumed_write.finish_ns);
        require_close(
            continued_drain.finish_ns - continued_frontier,
            resumed_drain.finish_ns - resumed_frontier,
            "compact GC checkpoint latency");
        require_equal(continued_drain.physical_bytes, resumed_drain.physical_bytes,
                      "compact GC checkpoint bytes");
        continued_frontier = continued_drain.finish_ns;
        resumed_frontier = resumed_drain.finish_ns;
    }
    compare_delta(
        before_uninterrupted,
        uninterrupted.device.stats(),
        before_recovered,
        recovered.stats());
    compare_image(
        recovered.persistent_image(),
        uninterrupted.device.persistent_image());
    require(
        recovered.stats().gc_runs > 0,
        "compact recovery continuation did not exercise GC");
}

void test_image_rejects_volatile_obsolete_and_geometry_drift() {
    const auto config = persistence_config();
    HbfController pending(config);
    pending.prepopulate_logical_pages({0});
    (void)pending.issue(write_request(
        "pending/write", 0.0, 0, config.device.page_size_bytes));
    require_throws(
        [&] { (void)pending.persistent_image(); }, "requires quiescent");

    auto seeded = build_seeded_device(config);
    auto image = seeded.device.persistent_image();
    image.version = 2;
    HbfController obsolete(config);
    require_throws(
        [&] { obsolete.restore_persistent_image(image); }, "version");
    image.version = 7;
    image.page_size_bytes *= 2;
    HbfController mismatched(config);
    require_throws(
        [&] { mismatched.restore_persistent_image(image); }, "geometry");
    require_throws(
        [&] { seeded.device.restore_persistent_image(image); }, "fresh device");
}

void test_canonical_file_round_trip_and_strict_tail() {
    const auto config = persistence_config();
    auto seeded = build_compact_seeded_device(config);
    const auto image = seeded.device.persistent_image();
    const auto unique = std::to_string(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const auto directory = std::filesystem::temp_directory_path() /
        ("hbfsim-persistent-image-test-" + unique);
    std::filesystem::create_directory(directory);
    const auto path = directory / "image.hbfstate";
    try {
        write_persistent_image_file(path, image);
        compare_image(read_persistent_image_file(path), image);
        require_throws(
            [&] { write_persistent_image_file(path, image); }, "already exists");
        {
            std::ofstream output(path, std::ios::app);
            output << "TRAILING\n";
        }
        require_throws(
            [&] { (void)read_persistent_image_file(path); }, "trailing");
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
}

void test_implicit_invalid_image_still_rejects_corruption() {
    const auto config = persistence_config();
    auto seeded = build_seeded_device(config);
    const auto original = seeded.device.persistent_image();
    auto bad_count = original;
    bad_count.state.blocks.front().invalid_pages++;
    HbfController count_device(config);
    require_throws([&] { count_device.restore_persistent_image(bad_count); },
                   "capacity");

    auto missing_live_page = original;
    require(!missing_live_page.state.materialized_pages.empty(), "empty seed");
    missing_live_page.state.materialized_pages.pop_back();
    HbfController missing_device(config);
    require_throws(
        [&] { missing_device.restore_persistent_image(missing_live_page); },
        "bitmap");

    auto obsolete_page = original;
    obsolete_page.state.materialized_pages.front().status = "invalid";
    HbfController obsolete_device(config);
    require_throws([&] { obsolete_device.restore_persistent_image(obsolete_page); },
                   "page status");
}

void test_sparse_retirement_bits_cross_words_and_restore() {
    auto config = persistence_config();
    config.device.blocks_per_plane = 1024;
    config.device.pages_per_block = 16;
    config.host.mapping_entries_per_page = 16;
    HbfController original(config);
    original.prepopulate_mutable_logical_page_range(63, 130);
    double frontier = 0;
    const std::vector<std::uint64_t> retired{63, 64, 65, 127, 128, 191, 192};
    for (unsigned round = 0; round < 2; ++round) {
        for (const auto lpn : retired) {
            frontier = original.issue(write_request(
                "sparse-retirement", frontier, lpn, config.device.page_size_bytes)).finish_ns;
        }
        frontier = original.drain_pending("checkpoint", frontier).finish_ns;
    }
    const auto image = original.persistent_image();
    std::vector<std::uint64_t> exported_retired;
    image.compact_image->retired_lpns.for_each_ordered(
        [&](std::uint64_t page) { exported_retired.push_back(page); });
    require(exported_retired == retired,
            "retirement bits lost boundaries, duplicated rewrites, or changed order");
    HbfController restored(config);
    restored.restore_persistent_image(image);
    compare_image(restored.persistent_image(), image);
    const auto next = original.issue(write_request(
        "after-restore", frontier, 129, config.device.page_size_bytes));
    const auto resumed = restored.issue(write_request(
        "after-restore", 0, 129, config.device.page_size_bytes));
    require_close(next.finish_ns - frontier, resumed.finish_ns,
                  "sparse retirement recovery latency");
    (void)original.drain_pending("checkpoint", next.finish_ns);
    (void)restored.drain_pending("checkpoint", resumed.finish_ns);
    compare_image(restored.persistent_image(), original.persistent_image());

    auto corrupt = image;
    corrupt.compact_image->retired_lpns.insert(std::numeric_limits<std::uint64_t>::max());
    HbfController invalid(config);
    require_throws([&] { invalid.restore_persistent_image(corrupt); },
                   "retired LPN");
}

} // namespace

int main() {
    test_quiescent_image_restores_exact_continuation();
    test_multichannel_restart_preserves_media_with_cold_hbm();
    test_compact_image_restores_exact_gc_continuation();
    test_image_rejects_volatile_obsolete_and_geometry_drift();
    test_canonical_file_round_trip_and_strict_tail();
    test_implicit_invalid_image_still_rejects_corruption();
    test_sparse_retirement_bits_cross_words_and_restore();
    return 0;
}
