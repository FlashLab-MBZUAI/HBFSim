#include "app/system_config.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using hbfsim::app::SystemConfigBuilder;
using hbfsim::host::MappingMode;

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

void test_engine_owns_only_physical_configuration() {
    require(SystemConfigBuilder::owns_key("hbm-capacity-bytes"),
            "HBM capacity is not engine-owned");
    require(SystemConfigBuilder::owns_key("hbf-program-ns"),
            "HBF timing is not engine-owned");
    require(SystemConfigBuilder::owns_key("hbf-thermal-enable"),
            "HBF thermal model is not engine-owned");
    require(SystemConfigBuilder::owns_key("hbf-thermal-start-state"),
            "HBF thermal start state is not engine-owned");
    require(SystemConfigBuilder::owns_key("hbf-thermal-neighbor-heat-c"),
            "HBF thermal neighbor heat is not engine-owned");
    require(SystemConfigBuilder::owns_key("external-backing-kind"),
            "external backing is not engine-owned");
    for (const auto* key : {
             "inference-offload-correction-enable",
             "inference-offload-correction-fixed-ns",
             "inference-offload-correction-ns-per-gib",
             "inference-offload-correction-read-only",
             "inference-offload-correction-max-caller-concurrency"}) {
        require(
            !SystemConfigBuilder::owns_key(key),
            std::string("removed inference correction key is still owned: ") +
                key);
    }
    require(SystemConfigBuilder::owns_key("hbf-external-direct-link-enable"),
            "direct lane opt-in is not engine-owned");
    require(SystemConfigBuilder::owns_key(
                "external-backing-request-segment-bytes"),
            "external transport segmentation is not engine-owned");
    require(SystemConfigBuilder::owns_key(
                "external-backing-media-read-queues"),
            "external media read queues are not engine-owned");
    require(SystemConfigBuilder::owns_key(
                "external-backing-media-write-queues"),
            "external media write queues are not engine-owned");
    require(!SystemConfigBuilder::owns_key("external-backing-read-queues"),
            "external media queue alias leaked into engine-owned keys");
    {
        SystemConfigBuilder builder;
        require_throws(
            [&] { builder.apply("external-backing-read-queues", "2"); },
            "external media queue alias was accepted");
    }
    require(SystemConfigBuilder::owns_key("base-die-link-read-bw"),
            "D2D timing is not engine-owned");
    require(SystemConfigBuilder::owns_key("hbm-service-quantum-bytes"),
            "HBM refresh-to-refresh delay is not engine-owned");
    for (const auto* key : {
             "scenarios", "line-size", "flat-hbm-bytes",
             "layer-buffer-bytes", "behavioral-promotion-threshold"}) {
        require(!SystemConfigBuilder::owns_key(key),
                std::string("policy/workload key leaked into engine: ") + key);
        SystemConfigBuilder builder;
        require_throws(
            [&] { builder.apply(key, "1"); },
            std::string("engine accepted policy/workload key: ") + key);
    }
}

void test_hbm_channel_timing_is_configurable() {
    SystemConfigBuilder builder;
    builder.apply("hbm-read-latency-ns", "50");
    builder.apply("hbm-write-latency-ns", "40");
    builder.apply("hbm-bandwidth-efficiency", "0.8");
    builder.apply("hbm-service-quantum-bytes", "2048");
    builder.apply("hbm-service-group-channels", "4");
    const auto config = builder.resolve();
    require(config.hbm.timing.read_latency_ns == 50 &&
        config.hbm.timing.write_latency_ns == 40 &&
        config.hbm.timing.bandwidth_efficiency == 0.8 &&
        config.hbm.controller.service_quantum_bytes == 2048 &&
        config.hbm.controller.service_group_channels == 4,
        "HBM channel-model overrides were not preserved");
}

void test_capacity_derivation_and_mapping_memory() {
    SystemConfigBuilder builder;
    builder.apply("hbm-capacity-bytes", "1073741824");
    builder.apply("hbf-capacity-ratio", "4");
    builder.apply("hbf-stacks", "2");
    builder.apply("hbf-channels", "1");
    builder.apply("hbf-dies-per-channel", "1");
    builder.apply("hbf-planes-per-die", "2");
    builder.apply("hbf-pages-per-block", "8");
    builder.apply("hbf-page-size", "4096");
    const auto config = builder.resolve();
    const auto capacity =
        static_cast<std::uint64_t>(config.hbf.device.stacks) *
        config.hbf.device.channels_per_stack * config.hbf.device.dies_per_channel *
        config.hbf.device.planes_per_die * config.hbf.device.blocks_per_plane *
        config.hbf.device.pages_per_block * config.hbf.device.page_size_bytes;
    require(capacity >= 4ull * 1073741824ull,
            "derived HBF capacity is below its requested ratio");
    require(config.hbf.host.ctrl_dram_bytes > 0,
            "resident mapping memory was not derived");
}

void test_conflicting_capacity_contract_fails_closed() {
    SystemConfigBuilder builder;
    builder.apply("hbf-capacity-bytes", "4096");
    builder.apply("hbf-capacity-ratio", "2");
    require_throws(
        [&] { (void)builder.resolve(); },
        "conflicting HBF capacity contracts were accepted");
}

void test_resident_budget_includes_configured_write_buffer() {
    SystemConfigBuilder builder;
    builder.apply("hbf-stacks", "1");
    builder.apply("hbf-channels", "1");
    builder.apply("hbf-dies-per-channel", "1");
    builder.apply("hbf-planes-per-die", "1");
    builder.apply("hbf-blocks-per-plane", "8");
    builder.apply("hbf-pages-per-block", "8");
    builder.apply("hbf-page-size", "4096");
    builder.apply("hbf-write-coalescing", "true");
    builder.apply("hbf-write-buffer-pages", "3");
    const auto config = builder.resolve();
    require(
        config.hbf.host.ctrl_dram_bytes == 5 * 4096,
        "resident controller DRAM did not include one mapping page plus "
        "three write-buffer pages");
}

void test_cached_mapping_requires_and_preserves_explicit_budget() {
    SystemConfigBuilder missing_budget;
    missing_budget.apply("hbf-mapping-mode", "cached");
    require_throws(
        [&] { (void)missing_budget.resolve(); },
        "cached mapping accepted an implicit full-resident DRAM budget");

    SystemConfigBuilder cached;
    cached.apply("hbf-mapping-mode", "cached");
    cached.apply("hbf-ctrl-dram-bytes", "1048576");
    const auto config = cached.resolve();
    require(config.hbf.host.mapping_mode == MappingMode::Cached,
            "cached mapping mode was not preserved");
    require(config.hbf.host.ctrl_dram_bytes == 1048576,
            "cached mapping DRAM budget was changed during resolution");
}

void test_cached_mapping_derives_exact_capacity_fraction() {
    SystemConfigBuilder cached;
    cached.apply("hbf-mapping-mode", "cached");
    cached.apply("hbf-stacks", "2");
    cached.apply("hbf-channels", "1");
    cached.apply("hbf-dies-per-channel", "1");
    cached.apply("hbf-planes-per-die", "1");
    cached.apply("hbf-blocks-per-plane", "1000");
    cached.apply("hbf-pages-per-block", "8");
    cached.apply("hbf-page-size", "4096");
    cached.apply("hbf-ctrl-dram-capacity-denominator", "1000");
    const auto config = cached.resolve();
    require(config.hbf.host.ctrl_dram_capacity_denominator == 1000,
            "controller-DRAM denominator was not preserved");
    require(config.hbf.host.ctrl_dram_bytes == 2 * 8 * 4096,
            "controller-DRAM budget was not derived per stack");

    SystemConfigBuilder conflict;
    conflict.apply("hbf-mapping-mode", "cached");
    conflict.apply("hbf-ctrl-dram-bytes", "1048576");
    conflict.apply("hbf-ctrl-dram-capacity-denominator", "1000");
    require_throws(
        [&] { (void)conflict.resolve(); },
        "explicit and ratio-derived controller DRAM were both accepted");
}

void test_external_profile_and_overrides() {
    SystemConfigBuilder builder;
    builder.apply("external-backing-kind", "host-dram");
    builder.apply("external-backing-capacity-bytes", "1048576");
    builder.apply("external-backing-request-segment-bytes", "131072");
    builder.apply("external-backing-media-channels", "3");
    builder.apply("external-backing-media-read-queues", "2");
    builder.apply("external-backing-media-write-queues", "1");
    builder.apply("base-die-link-latency-ns", "17.5");
    const auto config = builder.resolve();
    require(
        config.external.kind ==
            hbfsim::physical::external::ExternalBackingKind::HostDram,
        "host DRAM profile identity was lost");
    require(config.external.capacity_bytes == 1048576,
            "external capacity override was lost");
    require(config.external.request_segment_bytes == 131072,
            "external request segment override was lost");
    require(config.external.media_channels == 3,
            "external channel override was lost");
    require(config.external.media_read_queues == 2,
            "external read queue override was lost");
    require(config.external.media_write_queues == 1,
            "external write queue override was lost");
    require(std::abs(config.base_die_link.latency_ns - 17.5) < 1e-12,
            "base-die link timing override was lost");

    require_throws(
        [&] { builder.apply("external-backing-kind", "dram"); },
        "ambiguous DRAM alias was accepted");
}

void test_integers_are_decimal_only() {
    SystemConfigBuilder builder;
    builder.apply("hbf-blocks-per-plane", "010");
    require(builder.resolve().hbf.device.blocks_per_plane == 10,
            "leading zero selected octal");
    for (const auto* raw : {"0x10", "-1", "+1", "1e3", " 7", "7 ", ""}) {
        SystemConfigBuilder rejecting;
        require_throws(
            [&] { rejecting.apply("hbf-blocks-per-plane", raw); },
            std::string("non-decimal integer was accepted: '") + raw + "'");
    }
}

void test_external_kind_cannot_change_after_overrides() {
    SystemConfigBuilder ordered;
    ordered.apply("external-backing-kind", "nvme-ssd");
    ordered.apply("external-backing-media-read-latency-ns", "12345");
    // Restating the same kind is a no-op.
    ordered.apply("external-backing-kind", "nvme-ssd");
    const auto config = ordered.resolve();
    require(
        config.external.kind ==
                hbfsim::physical::external::ExternalBackingKind::NvmeSsd &&
            std::abs(config.external.media_read_latency_ns - 12345.0) < 1e-9,
        "same-kind restatement lost overrides");
    require_throws(
        [&] { ordered.apply("external-backing-kind", "cxl-memory"); },
        "kind change after kind-specific overrides produced a hybrid device");

    SystemConfigBuilder implicit;
    implicit.apply("external-backing-capacity-bytes", "1048576");
    implicit.apply("external-backing-kind", "nvme-ssd");
    require(implicit.resolve().external.capacity_bytes == 1048576,
            "override before the first explicit kind was lost");
}

void test_direct_lane_is_opt_in_with_a_complete_envelope() {
    SystemConfigBuilder defaults;
    require(!defaults.resolve().hbf_external_direct_link.has_value(),
            "direct lane existed without opting in");

    SystemConfigBuilder partial;
    partial.apply("hbf-external-direct-link-enable", "true");
    partial.apply("hbf-external-direct-link-read-bw", "32");
    require_throws(
        [&] { (void)partial.resolve(); },
        "enabled direct lane resolved without a complete envelope");

    SystemConfigBuilder timing_only;
    timing_only.apply("hbf-external-direct-link-latency-ns", "100");
    require_throws(
        [&] { (void)timing_only.resolve(); },
        "direct lane timing was accepted without the opt-in");

    SystemConfigBuilder enabled;
    enabled.apply("hbf-external-direct-link-enable", "true");
    enabled.apply("hbf-external-direct-link-read-bw", "32");
    enabled.apply("hbf-external-direct-link-write-bw", "16");
    enabled.apply("hbf-external-direct-link-latency-ns", "100");
    const auto lane = enabled.resolve().hbf_external_direct_link;
    require(
        lane && std::abs(lane->read_bandwidth_GBps - 32.0) < 1e-12 &&
            std::abs(lane->write_bandwidth_GBps - 16.0) < 1e-12 &&
            std::abs(lane->latency_ns - 100.0) < 1e-12,
        "explicit direct lane envelope was not preserved");
}

void test_thermal_start_state_vocabulary() {
    SystemConfigBuilder builder;
    builder.apply("hbf-thermal-enable", "true");
    builder.apply("hbf-thermal-start-state", "throttle-ceiling");
    require(builder.resolve().hbf.device.thermal_start_at_ceiling,
            "throttle-ceiling start state was lost");
    builder.apply("hbf-thermal-start-state", "idle");
    require(!builder.resolve().hbf.device.thermal_start_at_ceiling,
            "idle start state was lost");
    require_throws(
        [&] { builder.apply("hbf-thermal-start-state", "hot"); },
        "engine accepted an unknown thermal start state");
}

void test_static_wear_leveling_configuration() {
    SystemConfigBuilder builder;
    for (const auto* key : {
             "hbf-static-wear-leveling-stop-gap",
             "hbf-static-wear-leveling-cooldown-erases",
             "hbf-static-wear-leveling-max-write-fraction"}) {
        require(SystemConfigBuilder::owns_key(key),
                std::string("wear-leveling control is not engine-owned: ") + key);
    }
    builder.apply("hbf-static-wear-leveling-erase-gap", "8");
    builder.apply("hbf-static-wear-leveling-stop-gap", "4");
    builder.apply("hbf-static-wear-leveling-cooldown-erases", "4294967297");
    builder.apply("hbf-static-wear-leveling-max-write-fraction", "0.125");
    const auto config = builder.resolve();
    require(config.hbf.host.static_wear_leveling_stop_gap == 4 &&
                config.hbf.host.static_wear_leveling_cooldown_erases == 4294967297ULL &&
                config.hbf.host.static_wear_leveling_max_write_fraction == .125,
            "wear-leveling controls were lost or narrowed during resolution");
    for (const auto fraction : {
             -.01, 1.01, std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN()}) {
        auto invalid = config.hbf;
        invalid.host.static_wear_leveling_max_write_fraction = fraction;
        require_throws(
            [&] { hbfsim::host::HbfController device(invalid); },
            "invalid wear-leveling write fraction was accepted");
    }
    auto invalid = config.hbf;
    invalid.host.static_wear_leveling_stop_gap = 9;
    require_throws(
        [&] { hbfsim::host::HbfController device(invalid); },
        "wear-leveling stop gap above the activation gap was accepted");
}

void test_superblock_planes_all_resolves_to_the_plane_count() {
    SystemConfigBuilder builder;
    builder.apply("hbf-stacks", "2");
    builder.apply("hbf-channels", "3");
    builder.apply("hbf-dies-per-channel", "1");
    builder.apply("hbf-planes-per-die", "2");
    builder.apply("hbf-mapping-organization", "extent");
    builder.apply("hbf-mapping-superblock-planes", "all");
    require(builder.resolve().hbf.host.mapping_superblock_planes == 12,
            "superblock-planes=all did not resolve to stacks x channels x dies x planes");
    // The width follows the final geometry, not the geometry at the time
    // 'all' was applied.
    builder.apply("hbf-planes-per-die", "4");
    require(builder.resolve().hbf.host.mapping_superblock_planes == 24,
            "superblock-planes=all did not track a later geometry change");
    // A later explicit width replaces 'all'.
    builder.apply("hbf-mapping-superblock-planes", "6");
    require(builder.resolve().hbf.host.mapping_superblock_planes == 6,
            "an explicit superblock width did not replace 'all'");
    require_throws(
        [&] { builder.apply("hbf-mapping-superblock-planes", "some"); },
        "a non-numeric superblock width other than 'all' was accepted");
}

} // namespace

int main() {
    try {
        test_engine_owns_only_physical_configuration();
        test_hbm_channel_timing_is_configurable();
        test_capacity_derivation_and_mapping_memory();
        test_conflicting_capacity_contract_fails_closed();
        test_resident_budget_includes_configured_write_buffer();
        test_cached_mapping_requires_and_preserves_explicit_budget();
        test_cached_mapping_derives_exact_capacity_fraction();
        test_external_profile_and_overrides();
        test_integers_are_decimal_only();
        test_external_kind_cannot_change_after_overrides();
        test_direct_lane_is_opt_in_with_a_complete_envelope();
        test_thermal_start_state_vocabulary();
        test_static_wear_leveling_configuration();
        test_superblock_planes_all_resolves_to_the_plane_count();
        std::cout << "system config ownership regression passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "system config ownership regression failed: "
                  << error.what() << '\n';
        return 1;
    }
}
