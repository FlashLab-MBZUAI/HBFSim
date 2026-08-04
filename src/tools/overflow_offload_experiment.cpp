#include "physical/hybrid/capacity_overflow_composition.hpp"
#include "tools/sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using hbfsim::physical::external::ExternalBackingConfig;
using hbfsim::physical::external::cxl_memory_profile;
using hbfsim::physical::external::nvme_ssd_profile;
using hbfsim::physical::hbf::HbfConfig;
using hbfsim::physical::hbm::HbmConfig;
using hbfsim::physical::hybrid::BaseDieLinkConfig;
using hbfsim::physical::hybrid::BackingTier;
using hbfsim::physical::hybrid::CapacityOverflowComposition;
using hbfsim::physical::hybrid::CapacityOverflowConfig;
using hbfsim::physical::hybrid::CapacityOverflowRunResult;
using hbfsim::physical::hybrid::CapacityOverflowWorkload;
using hbfsim::physical::hybrid::CapacityReadbackDestination;
using hbfsim::tools::sha256_file;
using hbfsim::tools::sha256_text;

constexpr std::uint64_t kPageSize = 4096;
constexpr std::uint64_t kMiB = 1ull << 20;
constexpr std::uint64_t kGiB = 1ull << 30;
constexpr std::uint64_t kDefaultHbmCapacityBytes = 192ull * kGiB;
constexpr std::uint64_t kDefaultTotalWriteBytes = 200ull * kGiB;
constexpr std::uint64_t kDefaultReadBufferBytes = 1ull * kMiB;
// Llama 3.1 405B transformer-block weights at 8 bit:
//   2*h*h + 2*h*(kv_heads*head_dim) + 3*h*intermediate + 2*h
// = 3,187,703,808 B for h=16,384, kv_heads=8, head_dim=128, and
// intermediate=53,248. Round the full block up to the 1 MiB transfer batch.
constexpr std::uint64_t kDefaultLayerLogicalBytes = 3'187'703'808ull;
constexpr std::uint64_t kDefaultLayerBytes = 3041ull * kMiB;
constexpr std::uint32_t kSampleHbfBlocksPerPlane = 2;
constexpr std::uint32_t kTargetHbfBlocksPerPlane = 8192;
constexpr double kConvergenceThreshold = 0.01;
constexpr std::array<std::uint64_t, 5> kServiceCurveWindows = {
    1, 4, 16, 64, 256,
};
constexpr std::array<std::uint64_t, 6> kLifecycleRestoreCounts = {
    1, 2, 4, 8, 16, 32,
};
constexpr std::uint64_t kServiceCurveResidentPages = 4096;
constexpr std::uint64_t kServiceCurveOffloadPages = 1024;
constexpr std::uint64_t kServiceCurveReadBufferPages = 256;

struct Options {
    std::uint64_t hbm_capacity_bytes = kDefaultHbmCapacityBytes;
    std::uint64_t total_write_bytes = kDefaultTotalWriteBytes;
    std::uint64_t read_buffer_bytes = kDefaultReadBufferBytes;
    std::uint64_t layer_bytes = kDefaultLayerBytes;
    std::size_t batch_pages = 256;
    std::size_t sample_batches = 32;
    std::filesystem::path json_path =
        "out/capacity-overflow/summary.json";
    std::filesystem::path csv_path =
        "out/capacity-overflow/summary.csv";
};

struct ExecutableIdentity {
    std::uint64_t bytes = 0;
    std::string sha256;
};

struct Geometry {
    std::uint64_t hbm_capacity_bytes = 0;
    std::uint64_t hbm_data_bytes = 0;
    std::uint64_t total_write_bytes = 0;
    std::uint64_t offload_bytes = 0;
    std::uint64_t read_buffer_bytes = 0;
    std::uint64_t hbm_data_pages = 0;
    std::uint64_t total_write_pages = 0;
    std::uint64_t offload_pages = 0;
    std::uint64_t read_buffer_pages = 0;
    std::uint64_t batch_pages = 0;
    std::uint64_t batch_bytes = 0;
    std::uint64_t target_batches = 0;
    std::uint64_t layer_bytes = 0;
    std::uint64_t layer_pages = 0;
    std::uint64_t layer_batches = 0;
    std::uint64_t layer_logical_bytes = 0;
    std::uint64_t layer_padding_bytes = 0;
    bool default_layer_sizing = false;
    std::uint64_t sample_resident_pages = 0;
    std::uint64_t doubled_sample_resident_pages = 0;
    std::uint64_t base_sample_batches = 0;
    std::uint64_t doubled_sample_batches = 0;
};

struct LayerRoundTripProjection {
    CapacityOverflowRunResult base_sample;
    CapacityOverflowRunResult doubled_sample;
    double offload_ns_per_batch_base = 0.0;
    double offload_ns_per_batch_doubled = 0.0;
    double restore_ns_per_batch_base = 0.0;
    double restore_ns_per_batch_doubled = 0.0;
    double offload_relative_drift = 0.0;
    double restore_relative_drift = 0.0;
    double projected_offload_elapsed_ns = 0.0;
    double projected_restore_elapsed_ns = 0.0;

    [[nodiscard]] double projected_e2e_elapsed_ns() const {
        return projected_offload_elapsed_ns +
            projected_restore_elapsed_ns;
    }
};

struct ServiceCurvePoint {
    std::uint64_t window_pages = 0;
    CapacityOverflowRunResult sample;
};

struct ProjectedCase {
    std::string name;
    BackingTier backing = BackingTier::Hbf;
    CapacityOverflowRunResult base_sample;
    CapacityOverflowRunResult doubled_sample;
    double offload_ns_per_batch_base = 0.0;
    double offload_ns_per_batch_doubled = 0.0;
    double read_ns_per_batch_base = 0.0;
    double read_ns_per_batch_doubled = 0.0;
    double fill_ns_per_byte_base = 0.0;
    double fill_ns_per_byte_doubled = 0.0;
    double offload_relative_drift = 0.0;
    double read_relative_drift = 0.0;
    double fill_relative_drift = 0.0;
    double projected_fill_elapsed_ns = 0.0;
    double projected_offload_elapsed_ns = 0.0;
    double projected_read_elapsed_ns = 0.0;
    double projected_drain_tail_ns = 0.0;
    std::uint64_t target_mapping_programs = 0;
    std::uint64_t target_mapping_checkpoint_rounds = 0;
    LayerRoundTripProjection layer_round_trip;
    std::vector<ServiceCurvePoint> service_curve;
};

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

std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw std::runtime_error("division by zero");
    }
    return numerator / denominator +
        static_cast<std::uint64_t>(numerator % denominator != 0);
}

std::uint64_t parse_u64(const std::string& value, const char* name) {
    std::size_t used = 0;
    const auto parsed = std::stoull(value, &used, 0);
    if (used != value.size()) {
        throw std::runtime_error(
            std::string(name) + " has trailing characters");
    }
    return parsed;
}

std::size_t parse_size(const std::string& value, const char* name) {
    const auto parsed = parse_u64(value, name);
    if (parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds size_t");
    }
    return static_cast<std::size_t>(parsed);
}

void usage(const char* argv0) {
    std::cerr
        << "usage: " << argv0 << " [options]\n"
        << "  --hbm-capacity-bytes N  physical HBM capacity (default 192 GiB)\n"
        << "  --total-write-bytes N   saturated KV writes (default 200 GiB)\n"
        << "  --read-buffer-bytes N   reserved HBM DMA buffer (default 1 MiB)\n"
        << "  --layer-bytes N         batch-aligned layer footprint "
           "(default 3041 MiB)\n"
        << "  --batch-pages N         offloading/readback batch (default 256 pages)\n"
        << "  --sample-batches N      base exact replay window (default 32)\n"
        << "  --json PATH             output JSON artifact\n"
        << "  --csv PATH              output CSV table\n";
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--hbm-capacity-bytes") {
            options.hbm_capacity_bytes = parse_u64(
                need("--hbm-capacity-bytes"),
                "--hbm-capacity-bytes");
        } else if (arg == "--total-write-bytes") {
            options.total_write_bytes = parse_u64(
                need("--total-write-bytes"),
                "--total-write-bytes");
        } else if (arg == "--read-buffer-bytes") {
            options.read_buffer_bytes = parse_u64(
                need("--read-buffer-bytes"),
                "--read-buffer-bytes");
        } else if (arg == "--layer-bytes") {
            options.layer_bytes = parse_u64(
                need("--layer-bytes"),
                "--layer-bytes");
        } else if (arg == "--batch-pages") {
            options.batch_pages = parse_size(
                need("--batch-pages"),
                "--batch-pages");
        } else if (arg == "--sample-batches") {
            options.sample_batches = parse_size(
                need("--sample-batches"),
                "--sample-batches");
        } else if (arg == "--json") {
            options.json_path = need("--json");
        } else if (arg == "--csv") {
            options.csv_path = need("--csv");
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

HbmConfig make_hbm_config(std::uint64_t capacity_bytes) {
    HbmConfig config;
    config.capacity_bytes = capacity_bytes;
    config.stacks = 4;
    config.channels_per_stack = 32;
    config.pseudo_channels_per_channel = 2;
    config.bank_groups_per_pseudo_channel = 16;
    config.banks_per_group = 4;
    config.channel_row_size_bytes = 2048;
    config.channel_width_bits = 64;
    config.burst_length = 8;
    config.pin_rate_Gbps = 6.4;
    config.data_rate_per_command_clock = 4;
    config.tCCD_S_cycles = 2;
    config.tCCD_L_cycles = 4;
    return config;
}

HbfConfig make_hbf_config(std::uint32_t blocks_per_plane) {
    HbfConfig config;
    config.stacks = 4;
    config.channels_per_stack = 4;
    config.dies_per_channel = 4;
    config.planes_per_die = 4;
    config.blocks_per_plane = blocks_per_plane;
    config.pages_per_block = 256;
    config.page_size_bytes = kPageSize;
    config.oob_bytes_per_page = 224;
    config.media_lanes_per_plane = 16;
    config.subarrays_per_plane = 32;
    config.page_buffer_banks_per_plane = 16;
    config.t_read_page_ns = 1000.0;
    config.t_program_page_ns = 95000.0;
    config.t_program_verify_ns = 5000.0;
    config.t_erase_block_ns = 2'000'000.0;
    config.ecc_decode_latency_ns = 250.0;
    config.ecc_encode_latency_ns = 250.0;
    config.ecc_decode_raw_bandwidth_GBps_per_die = 105.46875;
    config.ecc_encode_raw_bandwidth_GBps_per_die = 105.46875;
    config.channel_bandwidth_GBps = 421.875;
    config.hb_io_bandwidth_GBps = 1600.0;
    config.tsv_bandwidth_GBps = 1712.5;
    config.media_lane_bandwidth_GBps = 2048.0;
    config.page_buffer_bandwidth_GBps = 2048.0;
    config.read_buffer_pages = 0;
    config.write_coalescing_enabled = false;
    // Victim reuse waits for the payload page program, not an SRAM ack.
    config.write_buffer_completion_requires_flush = true;
    config.write_buffer_pages = 256;
    config.write_buffer_flush_threshold_pages = 128;
    // The target writes only ~0.4% of a fresh 2 TiB HBF, so this experiment
    // deliberately excludes steady-state GC from both sample and projection.
    config.gc_low_watermark_pages = 0;
    config.gc_hard_watermark_pages = 0;
    config.gc_reserved_free_blocks_per_plane = 0;
    return config;
}

BaseDieLinkConfig make_base_die_link() {
    return BaseDieLinkConfig{
        .read_bandwidth_GBps = 2048.0,
        .write_bandwidth_GBps = 512.0,
        .latency_ns = 0.0,
    };
}

std::uint64_t hbf_capacity_bytes(const HbfConfig& config) {
    auto pages = checked_mul(
        config.stacks,
        config.channels_per_stack,
        "HBF stack/channel pages");
    pages = checked_mul(pages, config.dies_per_channel, "HBF die pages");
    pages = checked_mul(pages, config.planes_per_die, "HBF plane pages");
    pages = checked_mul(pages, config.blocks_per_plane, "HBF block pages");
    pages = checked_mul(pages, config.pages_per_block, "HBF page capacity");
    return checked_mul(pages, config.page_size_bytes, "HBF byte capacity");
}

Geometry derive_geometry(const Options& options) {
    if (options.hbm_capacity_bytes == 0 ||
        options.total_write_bytes == 0 ||
        options.read_buffer_bytes == 0 ||
        options.layer_bytes == 0 ||
        options.batch_pages == 0 ||
        options.sample_batches == 0 ||
        options.hbm_capacity_bytes % kPageSize != 0 ||
        options.total_write_bytes % kPageSize != 0 ||
        options.read_buffer_bytes % kPageSize != 0 ||
        options.layer_bytes % kPageSize != 0 ||
        options.read_buffer_bytes >= options.hbm_capacity_bytes) {
        throw std::runtime_error(
            "capacities must be positive, page aligned, and leave HBM data");
    }
    if (options.total_write_bytes <= options.hbm_capacity_bytes) {
        throw std::runtime_error(
            "total writes must directly exceed physical HBM capacity");
    }

    Geometry geometry;
    geometry.hbm_capacity_bytes = options.hbm_capacity_bytes;
    geometry.hbm_data_bytes =
        options.hbm_capacity_bytes - options.read_buffer_bytes;
    geometry.total_write_bytes = options.total_write_bytes;
    geometry.offload_bytes =
        options.total_write_bytes - geometry.hbm_data_bytes;
    geometry.read_buffer_bytes = options.read_buffer_bytes;
    geometry.hbm_data_pages = geometry.hbm_data_bytes / kPageSize;
    geometry.total_write_pages = options.total_write_bytes / kPageSize;
    geometry.offload_pages = geometry.offload_bytes / kPageSize;
    geometry.read_buffer_pages = options.read_buffer_bytes / kPageSize;
    geometry.batch_pages = options.batch_pages;
    geometry.batch_bytes = checked_mul(
        geometry.batch_pages,
        kPageSize,
        "transfer batch bytes");
    geometry.layer_bytes = options.layer_bytes;
    geometry.layer_pages = options.layer_bytes / kPageSize;
    geometry.default_layer_sizing =
        options.layer_bytes == kDefaultLayerBytes;
    geometry.layer_logical_bytes =
        geometry.default_layer_sizing ?
        kDefaultLayerLogicalBytes :
        options.layer_bytes;
    geometry.layer_padding_bytes =
        geometry.layer_bytes - geometry.layer_logical_bytes;
    if (geometry.batch_pages > geometry.hbm_data_pages ||
        geometry.batch_pages > geometry.read_buffer_pages ||
        geometry.offload_pages % geometry.batch_pages != 0 ||
        geometry.layer_pages % geometry.batch_pages != 0) {
        throw std::runtime_error(
            "batch must fit both HBM regions and exactly divide overflow "
            "and layer");
    }
    geometry.target_batches =
        geometry.offload_pages / geometry.batch_pages;
    geometry.layer_batches =
        geometry.layer_pages / geometry.batch_pages;
    geometry.base_sample_batches = options.sample_batches;
    geometry.doubled_sample_batches = checked_mul(
        options.sample_batches,
        2,
        "doubled sample batches");
    geometry.sample_resident_pages = std::max<std::uint64_t>(
        4096,
        checked_mul(
            geometry.batch_pages,
            16,
            "sample resident pages"));
    geometry.doubled_sample_resident_pages = checked_mul(
        geometry.sample_resident_pages,
        2,
        "doubled sample resident pages");
    if (geometry.layer_bytes > geometry.offload_bytes ||
        geometry.layer_bytes > geometry.hbm_data_bytes) {
        throw std::runtime_error(
            "layer must fit both the offloaded set and HBM resident window");
    }
    if (geometry.doubled_sample_resident_pages > geometry.hbm_data_pages ||
        geometry.doubled_sample_batches > geometry.target_batches ||
        geometry.doubled_sample_batches > geometry.layer_batches) {
        throw std::runtime_error(
            "target HBM, offloading volume, and layer must contain the "
            "doubled replay windows");
    }

    const auto sample_hbf = make_hbf_config(kSampleHbfBlocksPerPlane);
    const auto doubled_sample_pages = checked_mul(
        geometry.doubled_sample_batches,
        geometry.batch_pages,
        "doubled sample overflow pages");
    const auto doubled_sample_bytes = checked_mul(
        doubled_sample_pages,
        kPageSize,
        "doubled sample overflow bytes");
    if (checked_mul(
            doubled_sample_bytes,
            2,
            "sample HBF headroom bytes") >
        hbf_capacity_bytes(sample_hbf)) {
        throw std::runtime_error(
            "sample window is too large for the sampled HBF geometry");
    }
    const auto target_hbf = make_hbf_config(kTargetHbfBlocksPerPlane);
    if (checked_mul(
            geometry.offload_bytes,
            2,
            "target HBF headroom bytes") >
        hbf_capacity_bytes(target_hbf)) {
        throw std::runtime_error(
            "target offloading is too large for fresh-media HBF projection");
    }
    if (geometry.offload_bytes > cxl_memory_profile().capacity_bytes ||
        geometry.offload_bytes > nvme_ssd_profile().capacity_bytes) {
        throw std::runtime_error(
            "target offloading exceeds an external backing capacity");
    }
    return geometry;
}

CapacityOverflowConfig make_sample_config(
    const Geometry& geometry,
    std::uint64_t resident_pages,
    BackingTier backing,
    ExternalBackingConfig external,
    CapacityReadbackDestination readback_destination =
        CapacityReadbackDestination::DmaBuffer,
    std::uint64_t transfer_window_pages = 0,
    std::uint64_t read_buffer_pages = 0) {
    if (transfer_window_pages == 0) {
        transfer_window_pages = geometry.batch_pages;
    }
    if (read_buffer_pages == 0) {
        read_buffer_pages = geometry.read_buffer_pages;
    }
    const auto sample_hbm_pages = checked_add(
        resident_pages,
        read_buffer_pages,
        "sample HBM pages");
    external.page_size_bytes = kPageSize;

    CapacityOverflowConfig config;
    config.hbm = make_hbm_config(checked_mul(
        sample_hbm_pages,
        kPageSize,
        "sample HBM bytes"));
    config.backing = backing;
    config.hbf = make_hbf_config(kSampleHbfBlocksPerPlane);
    config.external_backing = std::move(external);
    config.base_die_link = make_base_die_link();
    config.read_buffer_bytes = checked_mul(
        read_buffer_pages,
        kPageSize,
        "sample read-buffer bytes");
    config.transfer_batch_pages =
        static_cast<std::size_t>(transfer_window_pages);
    config.readback_destination = readback_destination;
    config.address_heatmap_bins = 256;
    config.trace = {.mode = hbfsim::physical::TraceMode::Off};
    return config;
}

CapacityOverflowRunResult run_sample(
    const Geometry& geometry,
    std::uint64_t resident_pages,
    BackingTier backing,
    ExternalBackingConfig external,
    std::uint64_t batches,
    CapacityReadbackDestination readback_destination =
        CapacityReadbackDestination::DmaBuffer) {
    const auto offload_pages = checked_mul(
        batches,
        geometry.batch_pages,
        "sample overflow pages");
    const CapacityOverflowWorkload workload{
        .base_addr = 0,
        .total_write_pages = checked_add(
            resident_pages,
            offload_pages,
            "sample write pages"),
        .first_arrival_ns = 0.0,
        .interarrival_ns = 0.0,
    };
    return CapacityOverflowComposition(make_sample_config(
        geometry,
        resident_pages,
        backing,
        std::move(external),
        readback_destination)).run(workload);
}

CapacityOverflowRunResult run_service_curve_sample(
    const Geometry& geometry,
    BackingTier backing,
    ExternalBackingConfig external,
    std::uint64_t window_pages) {
    if (window_pages == 0 ||
        window_pages > kServiceCurveReadBufferPages ||
        kServiceCurveOffloadPages % window_pages != 0) {
        throw std::runtime_error(
            "service-curve window does not divide the exact sample");
    }
    external.max_outstanding_requests =
        static_cast<std::uint32_t>(window_pages);
    const CapacityOverflowWorkload workload{
        .base_addr = 0,
        .total_write_pages = checked_add(
            kServiceCurveResidentPages,
            kServiceCurveOffloadPages,
            "service-curve write pages"),
        .first_arrival_ns = 0.0,
        .interarrival_ns = 0.0,
    };
    return CapacityOverflowComposition(make_sample_config(
        geometry,
        kServiceCurveResidentPages,
        backing,
        std::move(external),
        CapacityReadbackDestination::DmaBuffer,
        window_pages,
        kServiceCurveReadBufferPages)).run(workload);
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto position =
        (p / 100.0) * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(position));
    const auto hi = static_cast<std::size_t>(std::ceil(position));
    if (lo == hi) {
        return values[lo];
    }
    const auto fraction = position - static_cast<double>(lo);
    return values[lo] + (values[hi] - values[lo]) * fraction;
}

double throughput_GBps(std::uint64_t bytes, double elapsed_ns) {
    return elapsed_ns > 0.0 ?
        static_cast<double>(bytes) / elapsed_ns : 0.0;
}

double payload_wire_efficiency(
    std::uint64_t payload_bytes,
    std::uint64_t wire_bytes) {
    if (wire_bytes == 0 || payload_bytes > wire_bytes) {
        throw std::runtime_error(
            "payload/wire accounting is not a valid efficiency ratio");
    }
    return static_cast<double>(payload_bytes) /
        static_cast<double>(wire_bytes);
}

double relative_drift(double baseline, double candidate) {
    if (!std::isfinite(baseline) || !std::isfinite(candidate) ||
        baseline <= 0.0 || candidate <= 0.0) {
        throw std::runtime_error(
            "sample batch timing must be finite and positive");
    }
    return std::abs(candidate - baseline) / candidate;
}

std::uint64_t mapping_programs_for_prefix(
    std::uint64_t logical_pages,
    const HbfConfig& config) {
    std::uint64_t programs = 0;
    for (std::uint64_t stack = 0; stack < config.stacks; ++stack) {
        const auto local_pages = logical_pages / config.stacks +
            static_cast<std::uint64_t>(stack < logical_pages % config.stacks);
        programs = checked_add(
            programs,
            ceil_div(local_pages, config.mapping_entries_per_page),
            "mapping checkpoint programs");
    }
    return programs;
}

std::uint64_t checkpoint_rounds(
    std::uint64_t mapping_programs,
    const HbfConfig& config) {
    const auto planes = checked_mul(
        checked_mul(
            checked_mul(
                config.stacks,
                config.channels_per_stack,
                "HBF checkpoint stack/channels"),
            config.dies_per_channel,
            "HBF checkpoint dies"),
        config.planes_per_die,
        "HBF checkpoint planes");
    return ceil_div(mapping_programs, planes);
}

LayerRoundTripProjection project_layer_round_trip(
    const Geometry& geometry,
    const std::string& case_name,
    BackingTier backing,
    ExternalBackingConfig external) {
    LayerRoundTripProjection result;
    const auto base_resident_pages = checked_mul(
        geometry.base_sample_batches,
        geometry.batch_pages,
        "base layer sample resident pages");
    const auto doubled_resident_pages = checked_mul(
        geometry.doubled_sample_batches,
        geometry.batch_pages,
        "doubled layer sample resident pages");
    result.base_sample = run_sample(
        geometry,
        base_resident_pages,
        backing,
        external,
        geometry.base_sample_batches,
        CapacityReadbackDestination::OriginalSlots);
    result.doubled_sample = run_sample(
        geometry,
        doubled_resident_pages,
        backing,
        std::move(external),
        geometry.doubled_sample_batches,
        CapacityReadbackDestination::OriginalSlots);
    if (result.base_sample.stats.readback_destination !=
            CapacityReadbackDestination::OriginalSlots ||
        result.doubled_sample.stats.readback_destination !=
            CapacityReadbackDestination::OriginalSlots) {
        throw std::runtime_error(
            case_name + " layer replay did not restore original HBM slots");
    }

    result.offload_ns_per_batch_base =
        result.base_sample.stats.offload_elapsed_ns() /
        static_cast<double>(geometry.base_sample_batches);
    result.offload_ns_per_batch_doubled =
        result.doubled_sample.stats.offload_elapsed_ns() /
        static_cast<double>(geometry.doubled_sample_batches);
    result.restore_ns_per_batch_base =
        result.base_sample.stats.read_elapsed_ns() /
        static_cast<double>(geometry.base_sample_batches);
    result.restore_ns_per_batch_doubled =
        result.doubled_sample.stats.read_elapsed_ns() /
        static_cast<double>(geometry.doubled_sample_batches);
    result.offload_relative_drift = relative_drift(
        result.offload_ns_per_batch_base,
        result.offload_ns_per_batch_doubled);
    result.restore_relative_drift = relative_drift(
        result.restore_ns_per_batch_base,
        result.restore_ns_per_batch_doubled);
    if (result.offload_relative_drift > kConvergenceThreshold ||
        result.restore_relative_drift > kConvergenceThreshold) {
        throw std::runtime_error(
            case_name + " layer round-trip timing did not converge "
            "(offloading=" +
            std::to_string(result.offload_relative_drift) +
            ", restore=" +
            std::to_string(result.restore_relative_drift) + ")");
    }

    result.projected_offload_elapsed_ns =
        result.offload_ns_per_batch_doubled *
        static_cast<double>(geometry.layer_batches);
    result.projected_restore_elapsed_ns =
        result.restore_ns_per_batch_doubled *
        static_cast<double>(geometry.layer_batches);
    return result;
}

ProjectedCase project_case(
    const Geometry& geometry,
    std::string name,
    BackingTier backing,
    ExternalBackingConfig external) {
    ProjectedCase result;
    result.name = std::move(name);
    result.backing = backing;
    const auto layer_external = external;
    result.base_sample = run_sample(
        geometry,
        geometry.sample_resident_pages,
        backing,
        external,
        geometry.base_sample_batches);
    result.doubled_sample = run_sample(
        geometry,
        geometry.doubled_sample_resident_pages,
        backing,
        std::move(external),
        geometry.doubled_sample_batches);

    result.offload_ns_per_batch_base =
        result.base_sample.stats.offload_elapsed_ns() /
        static_cast<double>(geometry.base_sample_batches);
    result.offload_ns_per_batch_doubled =
        result.doubled_sample.stats.offload_elapsed_ns() /
        static_cast<double>(geometry.doubled_sample_batches);
    result.read_ns_per_batch_base =
        result.base_sample.stats.read_elapsed_ns() /
        static_cast<double>(geometry.base_sample_batches);
    result.read_ns_per_batch_doubled =
        result.doubled_sample.stats.read_elapsed_ns() /
        static_cast<double>(geometry.doubled_sample_batches);
    const auto base_sample_data_bytes = checked_mul(
        geometry.sample_resident_pages,
        kPageSize,
        "base sample resident bytes");
    const auto doubled_sample_data_bytes = checked_mul(
        geometry.doubled_sample_resident_pages,
        kPageSize,
        "doubled sample resident bytes");
    result.fill_ns_per_byte_base =
        result.base_sample.stats.fill_elapsed_ns() /
        static_cast<double>(base_sample_data_bytes);
    result.fill_ns_per_byte_doubled =
        result.doubled_sample.stats.fill_elapsed_ns() /
        static_cast<double>(doubled_sample_data_bytes);
    result.offload_relative_drift = relative_drift(
        result.offload_ns_per_batch_base,
        result.offload_ns_per_batch_doubled);
    result.read_relative_drift = relative_drift(
        result.read_ns_per_batch_base,
        result.read_ns_per_batch_doubled);
    result.fill_relative_drift = relative_drift(
        result.fill_ns_per_byte_base,
        result.fill_ns_per_byte_doubled);
    if (result.offload_relative_drift > kConvergenceThreshold ||
        result.read_relative_drift > kConvergenceThreshold ||
        result.fill_relative_drift > kConvergenceThreshold) {
        throw std::runtime_error(
            result.name + " periodic batch timing did not converge "
            "(fill=" + std::to_string(result.fill_relative_drift) +
            ", offloading=" + std::to_string(result.offload_relative_drift) +
            ", read=" + std::to_string(result.read_relative_drift) + ")");
    }

    result.projected_fill_elapsed_ns =
        static_cast<double>(geometry.hbm_data_bytes) *
        result.fill_ns_per_byte_doubled;
    result.projected_offload_elapsed_ns =
        result.offload_ns_per_batch_doubled *
        static_cast<double>(geometry.target_batches);
    result.projected_read_elapsed_ns =
        result.read_ns_per_batch_doubled *
        static_cast<double>(geometry.target_batches);

    if (backing == BackingTier::Hbf) {
        const auto target_hbf =
            make_hbf_config(kTargetHbfBlocksPerPlane);
        const auto sample_mapping_programs = mapping_programs_for_prefix(
            result.doubled_sample.stats.offload_pages,
            target_hbf);
        if (result.doubled_sample.hbf_stats.mapping_page_programs !=
            sample_mapping_programs) {
            throw std::runtime_error(
                "sample HBF mapping checkpoints do not conserve");
        }
        result.target_mapping_programs = mapping_programs_for_prefix(
            geometry.offload_pages,
            target_hbf);
        result.target_mapping_checkpoint_rounds = checkpoint_rounds(
            result.target_mapping_programs,
            target_hbf);
        const auto sample_rounds = checkpoint_rounds(
            sample_mapping_programs,
            target_hbf);
        result.projected_drain_tail_ns =
            result.doubled_sample.stats.drain_tail_ns() *
            static_cast<double>(result.target_mapping_checkpoint_rounds) /
            static_cast<double>(sample_rounds);
    }
    result.layer_round_trip = project_layer_round_trip(
        geometry,
        result.name,
        backing,
        layer_external);
    result.service_curve.reserve(kServiceCurveWindows.size());
    for (const auto window_pages : kServiceCurveWindows) {
        result.service_curve.push_back(ServiceCurvePoint{
            .window_pages = window_pages,
            .sample = run_service_curve_sample(
                geometry,
                backing,
                layer_external,
                window_pages),
        });
    }
    return result;
}

void ensure_parent(const std::filesystem::path& path) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
}

ExecutableIdentity executable_identity(const char* argv0) {
    std::error_code error;
    auto path = std::filesystem::absolute(argv0, error);
    if (error) {
        throw std::runtime_error(
            "cannot resolve overflow experiment executable path");
    }
    path = std::filesystem::weakly_canonical(path, error);
    if (error || !std::filesystem::is_regular_file(path, error) || error) {
        throw std::runtime_error(
            "overflow experiment executable is not a regular file");
    }
    const auto bytes = std::filesystem::file_size(path, error);
    if (error || bytes > std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error(
            "cannot determine overflow experiment executable size");
    }
    return ExecutableIdentity{
        .bytes = static_cast<std::uint64_t>(bytes),
        .sha256 = sha256_file(path),
    };
}

std::string sample_canonical_record(
    const CapacityOverflowRunResult& sample) {
    const auto& stats = sample.stats;
    const auto backing_read_bytes =
        stats.backing == BackingTier::Hbf ?
        sample.hbf_stats.logical_read_bytes :
        sample.external_backing_stats.read_bytes;
    const auto backing_write_bytes =
        stats.backing == BackingTier::Hbf ?
        sample.hbf_stats.logical_write_bytes :
        sample.external_backing_stats.write_bytes;
    const auto bits = [](double value) {
        std::ostringstream out;
        out << std::hex << std::setfill('0') << std::setw(16)
            << std::bit_cast<std::uint64_t>(value);
        return out.str();
    };
    std::ostringstream out;
    out << "hbfsim.capacity-overflow-sample.v1"
        << "|backing="
        << hbfsim::physical::hybrid::to_string(stats.backing)
        << "|hbm_data_pages=" << stats.hbm_data_pages
        << "|hbm_read_buffer_pages=" << stats.hbm_read_buffer_pages
        << "|transfer_batch_pages=" << stats.transfer_batch_pages
        << "|written_pages=" << stats.written_pages
        << "|offload_pages=" << stats.offload_pages
        << "|readback_pages=" << stats.readback_pages
        << "|offload_bytes=" << stats.offload_bytes
        << "|readback_bytes=" << stats.readback_bytes
        << "|fill_elapsed_bits=" << bits(stats.fill_elapsed_ns())
        << "|offload_elapsed_bits=" << bits(stats.offload_elapsed_ns())
        << "|read_elapsed_bits=" << bits(stats.read_elapsed_ns())
        << "|drain_tail_bits=" << bits(stats.drain_tail_ns())
        << "|hbm_read_bytes=" << sample.hbm_stats.read_bytes
        << "|hbm_write_bytes=" << sample.hbm_stats.write_bytes
        << "|backing_read_bytes=" << backing_read_bytes
        << "|backing_write_bytes=" << backing_write_bytes;
    return out.str();
}

void write_sample_summary_json(
    std::ostream& out,
    const CapacityOverflowRunResult& sample) {
    const auto record = sample_canonical_record(sample);
    const auto& stats = sample.stats;
    const auto backing_read_bytes =
        stats.backing == BackingTier::Hbf ?
        sample.hbf_stats.logical_read_bytes :
        sample.external_backing_stats.read_bytes;
    const auto backing_write_bytes =
        stats.backing == BackingTier::Hbf ?
        sample.hbf_stats.logical_write_bytes :
        sample.external_backing_stats.write_bytes;
    out << "{\"summary\":{\"backing\":\""
        << hbfsim::physical::hybrid::to_string(stats.backing)
        << "\",\"hbm_data_pages\":" << stats.hbm_data_pages
        << ",\"hbm_read_buffer_pages\":" << stats.hbm_read_buffer_pages
        << ",\"transfer_batch_pages\":" << stats.transfer_batch_pages
        << ",\"written_pages\":" << stats.written_pages
        << ",\"offload_pages\":" << stats.offload_pages
        << ",\"readback_pages\":" << stats.readback_pages
        << ",\"offload_bytes\":" << stats.offload_bytes
        << ",\"readback_bytes\":" << stats.readback_bytes
        << ",\"fill_elapsed_ns\":" << stats.fill_elapsed_ns()
        << ",\"offload_elapsed_ns\":" << stats.offload_elapsed_ns()
        << ",\"read_elapsed_ns\":" << stats.read_elapsed_ns()
        << ",\"drain_tail_ns\":" << stats.drain_tail_ns()
        << ",\"hbm_read_bytes\":" << sample.hbm_stats.read_bytes
        << ",\"hbm_write_bytes\":" << sample.hbm_stats.write_bytes
        << ",\"backing_read_bytes\":" << backing_read_bytes
        << ",\"backing_write_bytes\":" << backing_write_bytes
        << "},\"digest\":{\"algorithm\":\"sha256\",\"canonical_record\":\""
        << record << "\",\"value\":\"" << sha256_text(record) << "\"}}";
}

void write_profile_json(
    std::ostream& out,
    const Geometry& geometry) {
    const auto hbm = make_hbm_config(geometry.hbm_capacity_bytes);
    const auto sample_hbf = make_hbf_config(kSampleHbfBlocksPerPlane);
    const auto target_hbf = make_hbf_config(kTargetHbfBlocksPerPlane);
    const auto link = make_base_die_link();
    const auto cxl_memory = cxl_memory_profile();
    const auto ssd = nvme_ssd_profile();

    out << "  \"profile\":{\"base\":\"usecase-4h4f exact HBM capacity and "
        << "timing/topology\","
        << "\"page_size_bytes\":" << kPageSize << ","
        << "\"hbm\":{\"capacity_bytes\":" << hbm.capacity_bytes
        << ",\"stacks\":" << hbm.stacks
        << ",\"channels_per_stack\":" << hbm.channels_per_stack
        << ",\"pseudo_channels_per_channel\":"
        << hbm.pseudo_channels_per_channel
        << ",\"bank_groups_per_pseudo_channel\":"
        << hbm.bank_groups_per_pseudo_channel
        << ",\"banks_per_group\":" << hbm.banks_per_group
        << ",\"channel_row_size_bytes\":" << hbm.channel_row_size_bytes
        << ",\"channel_width_bits\":" << hbm.channel_width_bits
        << ",\"burst_length\":" << hbm.burst_length
        << ",\"pin_rate_Gbps\":" << hbm.pin_rate_Gbps
        << ",\"data_rate_per_command_clock\":"
        << hbm.data_rate_per_command_clock
        << ",\"tCCD_S_cycles\":" << hbm.tCCD_S_cycles
        << ",\"tCCD_L_cycles\":" << hbm.tCCD_L_cycles << "},"
        << "\"hbf\":{\"target_capacity_bytes\":"
        << hbf_capacity_bytes(target_hbf)
        << ",\"sample_capacity_bytes\":" << hbf_capacity_bytes(sample_hbf)
        << ",\"target_blocks_per_plane\":" << target_hbf.blocks_per_plane
        << ",\"sample_blocks_per_plane\":" << sample_hbf.blocks_per_plane
        << ",\"stacks\":" << target_hbf.stacks
        << ",\"channels_per_stack\":" << target_hbf.channels_per_stack
        << ",\"dies_per_channel\":" << target_hbf.dies_per_channel
        << ",\"planes_per_die\":" << target_hbf.planes_per_die
        << ",\"pages_per_block\":" << target_hbf.pages_per_block
        << ",\"page_size_bytes\":" << target_hbf.page_size_bytes
        << ",\"oob_bytes_per_page\":" << target_hbf.oob_bytes_per_page
        << ",\"mapping_entries_per_page\":"
        << target_hbf.mapping_entries_per_page
        << ",\"read_page_ns\":" << target_hbf.t_read_page_ns
        << ",\"program_page_ns\":" << target_hbf.t_program_page_ns
        << ",\"program_verify_ns\":" << target_hbf.t_program_verify_ns
        << ",\"ecc_decode_ns\":" << target_hbf.ecc_decode_latency_ns
        << ",\"ecc_encode_ns\":" << target_hbf.ecc_encode_latency_ns
        << ",\"ecc_raw_GBps_per_die\":"
        << target_hbf.ecc_decode_raw_bandwidth_GBps_per_die
        << ",\"channel_GBps\":" << target_hbf.channel_bandwidth_GBps
        << ",\"hbio_GBps\":" << target_hbf.hb_io_bandwidth_GBps
        << ",\"tsv_GBps\":" << target_hbf.tsv_bandwidth_GBps
        << ",\"read_buffer_pages\":" << target_hbf.read_buffer_pages
        << ",\"write_buffer_pages\":" << target_hbf.write_buffer_pages
        << ",\"write_coalescing\":"
        << (target_hbf.write_coalescing_enabled ? "true" : "false")
        << ",\"payload_program_completion\":"
        << (target_hbf.write_buffer_completion_requires_flush ?
            "true" : "false")
        << "},"
        << "\"base_die_link\":{\"read_GBps_per_stack\":"
        << link.read_bandwidth_GBps
        << ",\"write_GBps_per_stack\":" << link.write_bandwidth_GBps
        << ",\"latency_ns\":" << link.latency_ns
        << ",\"evidence_grade\":\"literature_derived\""
        << ",\"source_id\":\"kaist_tcad_2026_0452\""
        << ",\"source_scope\":\"architecture-model-input-not-vendor-measurement\""
        << ",\"sandisk_public_d2d_measurement_available\":false"
        << ",\"sandisk_public_hbf_read_GBps_per_stack\":1600},"
        << "\"cxl_memory\":{\"kind\":\""
        << hbfsim::physical::external::to_string(cxl_memory.kind)
        << "\",\"capacity_bytes\":" << cxl_memory.capacity_bytes
        << ",\"page_size_bytes\":" << cxl_memory.page_size_bytes
        << ",\"media_channels\":" << cxl_memory.media_channels
        << ",\"max_outstanding_requests\":"
        << cxl_memory.max_outstanding_requests
        << ",\"controller_issue_ns\":" << cxl_memory.controller_issue_ns
        << ",\"controller_processing_ns\":"
        << cxl_memory.controller_processing_ns
        << ",\"media_read_latency_ns\":"
        << cxl_memory.media_read_latency_ns
        << ",\"media_write_latency_ns\":"
        << cxl_memory.media_write_latency_ns
        << ",\"media_read_GBps\":"
        << cxl_memory.media_read_bandwidth_GBps
        << ",\"media_write_GBps\":"
        << cxl_memory.media_write_bandwidth_GBps
        << ",\"m2s_GBps\":" << cxl_memory.m2s_bandwidth_GBps
        << ",\"s2m_GBps\":" << cxl_memory.s2m_bandwidth_GBps
        << ",\"one_way_propagation_ns\":"
        << cxl_memory.one_way_propagation_ns
        << ",\"command_bytes\":" << cxl_memory.command_bytes
        << ",\"completion_bytes\":" << cxl_memory.completion_bytes << "},"
        << "\"nvme_ssd\":{\"kind\":\""
        << hbfsim::physical::external::to_string(ssd.kind)
        << "\",\"capacity_bytes\":" << ssd.capacity_bytes
        << ",\"page_size_bytes\":" << ssd.page_size_bytes
        << ",\"media_channels\":" << ssd.media_channels
        << ",\"max_outstanding_requests\":"
        << ssd.max_outstanding_requests
        << ",\"controller_issue_ns\":" << ssd.controller_issue_ns
        << ",\"controller_processing_ns\":"
        << ssd.controller_processing_ns
        << ",\"media_read_latency_ns\":" << ssd.media_read_latency_ns
        << ",\"media_write_latency_ns\":" << ssd.media_write_latency_ns
        << ",\"media_read_GBps\":" << ssd.media_read_bandwidth_GBps
        << ",\"media_write_GBps\":" << ssd.media_write_bandwidth_GBps
        << ",\"m2s_GBps\":" << ssd.m2s_bandwidth_GBps
        << ",\"s2m_GBps\":" << ssd.s2m_bandwidth_GBps
        << ",\"one_way_propagation_ns\":"
        << ssd.one_way_propagation_ns
        << ",\"command_bytes\":" << ssd.command_bytes
        << ",\"completion_bytes\":" << ssd.completion_bytes << "}},\n";
}

void write_json(
    const Options& options,
    const Geometry& geometry,
    const std::vector<ProjectedCase>& cases,
    const ExecutableIdentity& executable) {
    ensure_parent(options.json_path);
    std::ofstream out(options.json_path);
    if (!out) {
        throw std::runtime_error(
            "cannot open JSON output: " + options.json_path.string());
    }
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema\":{\"name\":\"hbfsim.capacity-overflow-experiment\","
        << "\"version\":7},\n"
        << "  \"generator\":{\"version\":\"" << HBFSIM_VERSION
        << "\",\"git_commit\":\"" << HBFSIM_GIT_COMMIT
        << "\",\"git_dirty\":"
        << (HBFSIM_GIT_DIRTY ? "true" : "false")
        << ",\"build_type\":\"" << HBFSIM_BUILD_TYPE
        << "\",\"compiler_id\":\"" << HBFSIM_COMPILER_ID
        << "\",\"compiler_version\":\"" << HBFSIM_COMPILER_VERSION
        << "\",\"executable\":{\"algorithm\":\"sha256\",\"value\":\""
        << executable.sha256 << "\",\"bytes\":" << executable.bytes
        << "}},\n"
        << "  \"validation\":{\"status\":\"exploratory_unattached\","
        << "\"certificate\":null},\n"
        << "  \"provenance\":{\"capacity\":\"exact\","
        << "\"traffic\":\"exact\","
        << "\"base_and_doubled_samples\":\"measured_sample\","
        << "\"target_timing\":\"projected\","
        << "\"service_curves\":\"measured_sample\","
        << "\"layer_lifecycle\":\"derived\"},\n"
        << "  \"interpretation\":\"saturated append-only KV/write-back "
        << "state; not read-only model weights\",\n"
        << "  \"policy\":{\"name\":\"fifo-write-back-with-reserved-read-buffer\","
        << "\"hbf_completion\":\"payload-page-program-complete\","
        << "\"readback\":\"backing-read-to-HBM-DMA-buffer-then-HBM-read\"},\n"
        << "  \"method\":{\"name\":\"real-capacity-converged-periodic-replay\","
        << "\"capacity_accounting\":\"exact\","
        << "\"traffic_accounting\":\"exact\","
        << "\"timing\":\"extrapolated-from-page-exact-batches\","
        << "\"layer_timing\":\"extrapolated-from-original-slot-restoring-"
           "page-exact-batches\","
        << "\"convergence_threshold\":" << kConvergenceThreshold << ","
        << "\"base_sample_batches\":" << geometry.base_sample_batches << ","
        << "\"doubled_sample_batches\":"
        << geometry.doubled_sample_batches << ","
        << "\"base_sample_resident_pages\":"
        << geometry.sample_resident_pages << ","
        << "\"doubled_sample_resident_pages\":"
        << geometry.doubled_sample_resident_pages
        << "},\n";
    write_profile_json(out, geometry);
    out << "  \"workload\":{\"hbm_capacity_bytes\":"
        << geometry.hbm_capacity_bytes
        << ",\"hbm_data_bytes\":" << geometry.hbm_data_bytes
        << ",\"read_buffer_bytes\":" << geometry.read_buffer_bytes
        << ",\"total_write_bytes\":" << geometry.total_write_bytes
        << ",\"total_write_pages\":" << geometry.total_write_pages
        << ",\"offload_bytes\":" << geometry.offload_bytes
        << ",\"offload_pages\":" << geometry.offload_pages
        << ",\"batch_pages\":" << geometry.batch_pages
        << ",\"batch_bytes\":" << geometry.batch_bytes
        << ",\"target_batches\":" << geometry.target_batches
        << ",\"interarrival_ns\":0},\n"
        << "  \"layer_round_trip\":{\"name\":\"configurable-layer-payload\","
        << "\"sizing_basis\":\""
        << (geometry.default_layer_sizing ?
            "llama-3.1-405b-8bit-transformer-block" :
            "user-configured")
        << "\",\"logical_bytes\":" << geometry.layer_logical_bytes
        << ",\"transfer_bytes\":" << geometry.layer_bytes
        << ",\"padding_bytes\":" << geometry.layer_padding_bytes
        << ",\"default_reference\":{\"model\":\"Llama-3.1-405B\","
        << "\"precision_bytes_per_parameter\":1,"
        << "\"logical_bytes\":" << kDefaultLayerLogicalBytes
        << ",\"transfer_alignment_bytes\":" << geometry.batch_bytes
        << ",\"source_url\":\"https://huggingface.co/meta-llama/"
           "Llama-3.1-405B-Instruct/blob/main/config.json\"}"
        << ",\"pages\":" << geometry.layer_pages
        << ",\"batches\":" << geometry.layer_batches
        << ",\"payload_semantics\":\"size-reference-not-steady-state-weight-write\""
        << ",\"immediate_readback\":true"
        << ",\"destination\":\"original-hbm-slots\""
        << ",\"completion\":\"last-page-restored-and-foreground-HBM-read-complete\""
        << ",\"mapping_checkpoint_drain_included\":false},\n"
        << "  \"cases\":[\n";

    const auto expected_hbm_write_bytes = checked_add(
        geometry.total_write_bytes,
        geometry.offload_bytes,
        "target HBM write bytes");
    const auto expected_hbm_read_bytes = checked_mul(
        geometry.offload_bytes,
        2,
        "target HBM read bytes");
    const auto expected_hbm_user_accesses = checked_add(
        geometry.total_write_pages,
        geometry.offload_pages,
        "target HBM user accesses");
    const auto expected_hbm_background_accesses = checked_mul(
        geometry.offload_pages,
        2,
        "target HBM background accesses");

    for (std::size_t index = 0; index < cases.size(); ++index) {
        const auto& item = cases[index];
        const auto fill_finish = item.projected_fill_elapsed_ns;
        const auto offload_start = fill_finish;
        const auto write_finish =
            offload_start + item.projected_offload_elapsed_ns;
        const auto read_start = write_finish;
        const auto read_finish =
            read_start + item.projected_read_elapsed_ns;
        const auto quiescent_finish =
            read_finish + item.projected_drain_tail_ns;
        const auto& sample = item.doubled_sample;
        const auto& layer = item.layer_round_trip;

        out << "    {\n"
            << "      \"name\":\"" << item.name << "\",\n"
            << "      \"backing\":\""
            << hbfsim::physical::hybrid::to_string(item.backing)
            << "\",\n"
            << "      \"capacity\":{\"hbm_total_bytes\":"
            << geometry.hbm_capacity_bytes
            << ",\"hbm_data_bytes\":" << geometry.hbm_data_bytes
            << ",\"hbm_read_buffer_bytes\":" << geometry.read_buffer_bytes
            << "},\n"
            << "      \"traffic\":{\"written_pages\":"
            << geometry.total_write_pages
            << ",\"offload_pages\":" << geometry.offload_pages
            << ",\"readback_pages\":" << geometry.offload_pages
            << ",\"offload_bytes\":" << geometry.offload_bytes
            << ",\"readback_bytes\":" << geometry.offload_bytes << "},\n"
            << "      \"raw_samples\":{\"base\":";
        write_sample_summary_json(out, item.base_sample);
        out << ",\"doubled\":";
        write_sample_summary_json(out, sample);
        out << "},\n"
            << "      \"sampling\":{\"base_batches\":"
            << geometry.base_sample_batches
            << ",\"doubled_batches\":" << geometry.doubled_sample_batches
            << ",\"base_offload_pages\":"
            << item.base_sample.stats.offload_pages
            << ",\"doubled_offload_pages\":"
            << sample.stats.offload_pages
            << ",\"offload_ns_per_batch_base\":"
            << item.offload_ns_per_batch_base
            << ",\"offload_ns_per_batch_doubled\":"
            << item.offload_ns_per_batch_doubled
            << ",\"offload_relative_drift\":"
            << item.offload_relative_drift
            << ",\"read_ns_per_batch_base\":"
            << item.read_ns_per_batch_base
            << ",\"read_ns_per_batch_doubled\":"
            << item.read_ns_per_batch_doubled
            << ",\"read_relative_drift\":" << item.read_relative_drift
            << ",\"fill_ns_per_byte_base\":"
            << item.fill_ns_per_byte_base
            << ",\"fill_ns_per_byte_doubled\":"
            << item.fill_ns_per_byte_doubled
            << ",\"fill_relative_drift\":" << item.fill_relative_drift
            << ",\"convergence\":\"PASS\"},\n"
            << "      \"write_phase\":{\"fill_elapsed_ns\":"
            << item.projected_fill_elapsed_ns
            << ",\"offload_elapsed_ns\":"
            << item.projected_offload_elapsed_ns
            << ",\"total_elapsed_ns\":"
            << item.projected_fill_elapsed_ns +
                item.projected_offload_elapsed_ns
            << ",\"offload_throughput_GBps\":"
            << throughput_GBps(
                geometry.offload_bytes,
                item.projected_offload_elapsed_ns)
            << ",\"offered_latency_p50_ns\":"
            << percentile(
                sample.offload_offered_latencies_ns,
                50.0)
            << ",\"offered_latency_p95_ns\":"
            << percentile(
                sample.offload_offered_latencies_ns,
                95.0)
            << ",\"offered_latency_p99_ns\":"
            << percentile(
                sample.offload_offered_latencies_ns,
                99.0)
            << ",\"service_latency_p50_ns\":"
            << percentile(
                sample.offload_service_latencies_ns,
                50.0)
            << ",\"service_latency_p95_ns\":"
            << percentile(
                sample.offload_service_latencies_ns,
                95.0)
            << ",\"service_latency_p99_ns\":"
            << percentile(
                sample.offload_service_latencies_ns,
                99.0)
            << ",\"elapsed_timing_basis\":\"periodic-extrapolation\""
            << ",\"latency_basis\":\"doubled-page-exact-sample\"},\n"
            << "      \"read_phase\":{\"elapsed_ns\":"
            << item.projected_read_elapsed_ns
            << ",\"throughput_GBps\":"
            << throughput_GBps(
                geometry.offload_bytes,
                item.projected_read_elapsed_ns)
            << ",\"offered_latency_p50_ns\":"
            << percentile(
                sample.readback_offered_latencies_ns,
                50.0)
            << ",\"offered_latency_p95_ns\":"
            << percentile(
                sample.readback_offered_latencies_ns,
                95.0)
            << ",\"offered_latency_p99_ns\":"
            << percentile(
                sample.readback_offered_latencies_ns,
                99.0)
            << ",\"service_latency_p50_ns\":"
            << percentile(
                sample.readback_service_latencies_ns,
                50.0)
            << ",\"service_latency_p95_ns\":"
            << percentile(
                sample.readback_service_latencies_ns,
                95.0)
            << ",\"service_latency_p99_ns\":"
            << percentile(
                sample.readback_service_latencies_ns,
                99.0)
            << ",\"elapsed_timing_basis\":\"periodic-extrapolation\""
            << ",\"latency_basis\":\"doubled-page-exact-sample\"},\n"
            << "      \"layer_round_trip\":{\"offload_elapsed_ns\":"
            << layer.projected_offload_elapsed_ns
            << ",\"restore_to_hbm_elapsed_ns\":"
            << layer.projected_restore_elapsed_ns
            << ",\"e2e_elapsed_ns\":"
            << layer.projected_e2e_elapsed_ns()
            << ",\"round_trip_throughput_GBps\":"
            << throughput_GBps(
                checked_mul(
                    geometry.layer_bytes,
                    2,
                    "layer round-trip traffic bytes"),
                layer.projected_e2e_elapsed_ns())
            << ",\"timing_basis\":\"original-slot-periodic-extrapolation\""
            << ",\"base_batches\":" << geometry.base_sample_batches
            << ",\"doubled_batches\":"
            << geometry.doubled_sample_batches
            << ",\"offload_ns_per_batch_base\":"
            << layer.offload_ns_per_batch_base
            << ",\"offload_ns_per_batch_doubled\":"
            << layer.offload_ns_per_batch_doubled
            << ",\"offload_relative_drift\":"
            << layer.offload_relative_drift
            << ",\"restore_ns_per_batch_base\":"
            << layer.restore_ns_per_batch_base
            << ",\"restore_ns_per_batch_doubled\":"
            << layer.restore_ns_per_batch_doubled
            << ",\"restore_relative_drift\":"
            << layer.restore_relative_drift
            << ",\"convergence\":\"PASS\"},\n";

        out << "      \"service_curve\":{\"provenance\":\"measured_sample\","
            << "\"resident_pages\":" << kServiceCurveResidentPages
            << ",\"offload_pages\":" << kServiceCurveOffloadPages
            << ",\"read_buffer_pages\":" << kServiceCurveReadBufferPages
            << ",\"points\":[\n";
        for (std::size_t curve_index = 0;
             curve_index < item.service_curve.size();
             ++curve_index) {
            const auto& point = item.service_curve[curve_index];
            const auto& curve = point.sample;
            const auto& hbf_stats = curve.hbf_stats;
            const auto& external_stats = curve.external_backing_stats;
            out << "        {\"window_pages\":" << point.window_pages
                << ",\"offload_bytes\":" << curve.stats.offload_bytes
                << ",\"readback_bytes\":" << curve.stats.readback_bytes
                << ",\"offload_elapsed_ns\":"
                << curve.stats.offload_elapsed_ns()
                << ",\"readback_elapsed_ns\":"
                << curve.stats.read_elapsed_ns()
                << ",\"offload_payload_GBps\":"
                << throughput_GBps(
                    curve.stats.offload_bytes,
                    curve.stats.offload_elapsed_ns())
                << ",\"readback_payload_GBps\":"
                << throughput_GBps(
                    curve.stats.readback_bytes,
                    curve.stats.read_elapsed_ns())
                << ",\"offload_latency_ns\":{\"offered\":{\"p50\":"
                << percentile(
                    curve.offload_offered_latencies_ns,
                    50.0)
                << ",\"p95\":"
                << percentile(
                    curve.offload_offered_latencies_ns,
                    95.0)
                << ",\"p99\":"
                << percentile(
                    curve.offload_offered_latencies_ns,
                    99.0)
                << "},\"service\":{\"p50\":"
                << percentile(
                    curve.offload_service_latencies_ns,
                    50.0)
                << ",\"p95\":"
                << percentile(
                    curve.offload_service_latencies_ns,
                    95.0)
                << ",\"p99\":"
                << percentile(
                    curve.offload_service_latencies_ns,
                    99.0)
                << "}},\"readback_latency_ns\":{\"offered\":{\"p50\":"
                << percentile(
                    curve.readback_offered_latencies_ns,
                    50.0)
                << ",\"p95\":"
                << percentile(
                    curve.readback_offered_latencies_ns,
                    95.0)
                << ",\"p99\":"
                << percentile(
                    curve.readback_offered_latencies_ns,
                    99.0)
                << "},\"service\":{\"p50\":"
                << percentile(
                    curve.readback_service_latencies_ns,
                    50.0)
                << ",\"p95\":"
                << percentile(
                    curve.readback_service_latencies_ns,
                    95.0)
                << ",\"p99\":"
                << percentile(
                    curve.readback_service_latencies_ns,
                    99.0)
                << "}},\"resource_utilization\":{";
            if (item.backing == BackingTier::Hbf) {
                out << "\"hbf_media\":" << hbf_stats.media_utilization()
                    << ",\"base_die_read\":"
                    << curve.base_die_link_stats.read_utilization()
                    << ",\"base_die_write\":"
                    << curve.base_die_link_stats.write_utilization()
                    << ",\"external_controller\":null"
                    << ",\"external_media\":null"
                    << ",\"external_m2s\":null"
                    << ",\"external_s2m\":null";
            } else {
                out << "\"hbf_media\":null"
                    << ",\"base_die_read\":null"
                    << ",\"base_die_write\":null"
                    << ",\"external_controller\":"
                    << external_stats.controller_utilization()
                    << ",\"external_media\":"
                    << external_stats.media_utilization()
                    << ",\"external_m2s\":"
                    << external_stats.m2s_utilization()
                    << ",\"external_s2m\":"
                    << external_stats.s2m_utilization();
            }
            out << "},\"queue_wait_work_ns\":{"
                << "\"device_outstanding\":"
                << (item.backing == BackingTier::Hbf ?
                    0.0 :
                    external_stats.outstanding_wait_ns)
                << ",\"backing_ingress\":"
                << (item.backing == BackingTier::Hbf ?
                    hbf_stats.stage_work.ingress_queue_wait_ns :
                    0.0)
                << ",\"backing_scheduler\":"
                << (item.backing == BackingTier::Hbf ?
                    hbf_stats.stage_work.scheduler_queue_wait_ns :
                    0.0)
                << ",\"backing_ecc\":"
                << (item.backing == BackingTier::Hbf ?
                    hbf_stats.stage_work.ecc_queue_wait_ns :
                    0.0)
                << ",\"base_die_read\":"
                << (item.backing == BackingTier::Hbf ?
                    curve.base_die_link_stats.read_queue_wait_ns :
                    0.0)
                << ",\"base_die_write\":"
                << (item.backing == BackingTier::Hbf ?
                    curve.base_die_link_stats.write_queue_wait_ns :
                    0.0)
                << ",\"external_controller\":"
                << (item.backing == BackingTier::Hbf ?
                    0.0 :
                    external_stats.controller_queue_wait_ns)
                << ",\"external_media\":"
                << (item.backing == BackingTier::Hbf ?
                    0.0 :
                    external_stats.media_queue_wait_ns)
                << ",\"external_m2s\":"
                << (item.backing == BackingTier::Hbf ?
                    0.0 :
                    external_stats.m2s_queue_wait_ns)
                << ",\"external_s2m\":"
                << (item.backing == BackingTier::Hbf ?
                    0.0 :
                    external_stats.s2m_queue_wait_ns)
                << "},\"wire_efficiency\":{";
            if (item.backing == BackingTier::Hbf) {
                out << "\"model\":\"payload-only\","
                    << "\"m2s\":1,\"s2m\":1";
            } else {
                out << "\"model\":\"payload-plus-protocol\","
                    << "\"m2s\":"
                    << payload_wire_efficiency(
                        external_stats.m2s_payload_bytes,
                        external_stats.m2s_wire_bytes)
                    << ",\"s2m\":"
                    << payload_wire_efficiency(
                        external_stats.s2m_payload_bytes,
                        external_stats.s2m_wire_bytes);
            }
            out << "},\"active_media_channels\":"
                << (item.backing == BackingTier::Hbf ?
                    hbf_stats.active_channels :
                    external_stats.active_media_channels)
                << ",\"max_device_outstanding\":";
            if (item.backing == BackingTier::Hbf) {
                out << "null";
            } else {
                out << external_stats.max_device_outstanding;
            }
            out << "}"
                << (curve_index + 1 == item.service_curve.size() ?
                    "\n" :
                    ",\n");
        }
        out << "      ]},\n"
            << "      \"timeline\":{\"fill_finish_ns\":" << fill_finish
            << ",\"offload_start_ns\":" << offload_start
            << ",\"write_finish_ns\":" << write_finish
            << ",\"read_start_ns\":" << read_start
            << ",\"read_finish_ns\":" << read_finish
            << ",\"quiescent_finish_ns\":" << quiescent_finish
            << ",\"drain_tail_ns\":"
            << item.projected_drain_tail_ns << "},\n"
            << "      \"hbm\":{\"read_bytes\":"
            << expected_hbm_read_bytes
            << ",\"write_bytes\":" << expected_hbm_write_bytes
            << ",\"user_accesses\":" << expected_hbm_user_accesses
            << ",\"background_accesses\":"
            << expected_hbm_background_accesses << "},\n";

        if (item.backing == BackingTier::Hbf) {
            const auto target_page_programs = checked_add(
                geometry.offload_pages,
                item.target_mapping_programs,
                "target HBF programs");
            out << "      \"hbf\":{\"logical_read_bytes\":"
                << geometry.offload_bytes
                << ",\"logical_write_bytes\":" << geometry.offload_bytes
                << ",\"physical_read_bytes\":" << geometry.offload_bytes
                << ",\"physical_write_bytes\":"
                << checked_mul(
                    target_page_programs,
                    kPageSize,
                    "target HBF physical writes")
                << ",\"read_requests\":" << geometry.offload_pages
                << ",\"program_requests\":" << geometry.offload_pages
                << ",\"page_reads\":" << geometry.offload_pages
                << ",\"data_programs\":" << geometry.offload_pages
                << ",\"page_programs\":" << target_page_programs
                << ",\"mapping_programs\":"
                << item.target_mapping_programs
                << ",\"mapping_checkpoint_rounds\":"
                << item.target_mapping_checkpoint_rounds
                << ",\"gc_runs\":0,\"gc_relocations\":0,\"block_erases\":0,"
                << "\"representative_media_utilization\":"
                << sample.hbf_stats.media_utilization() << "},\n"
                << "      \"base_die_link\":{\"read_transfers\":"
                << geometry.offload_pages
                << ",\"write_transfers\":" << geometry.offload_pages
                << ",\"read_bytes\":" << geometry.offload_bytes
                << ",\"write_bytes\":" << geometry.offload_bytes
                << ",\"representative_read_utilization\":"
                << sample.base_die_link_stats.read_utilization()
                << ",\"representative_write_utilization\":"
                << sample.base_die_link_stats.write_utilization()
                << "},\n"
                << "      \"external\":null,\n";
        } else {
            const auto config =
                sample.external_backing_stats.kind ==
                    hbfsim::physical::external::ExternalBackingKind::CxlMemory ?
                cxl_memory_profile() :
                nvme_ssd_profile();
            const auto request_count = checked_mul(
                geometry.offload_pages,
                2,
                "projected external request count");
            const auto m2s_protocol_bytes = checked_mul(
                request_count,
                config.command_bytes,
                "projected external M2S protocol bytes");
            const auto s2m_protocol_bytes = checked_mul(
                request_count,
                config.completion_bytes,
                "projected external S2M protocol bytes");
            const auto m2s_wire_bytes = checked_add(
                geometry.offload_bytes,
                m2s_protocol_bytes,
                "projected external M2S wire bytes");
            const auto s2m_wire_bytes = checked_add(
                geometry.offload_bytes,
                s2m_protocol_bytes,
                "projected external S2M wire bytes");
            out << "      \"hbf\":null,\n"
                << "      \"base_die_link\":null,\n"
                << "      \"external\":{\"kind\":\""
                << hbfsim::physical::external::to_string(
                    sample.external_backing_stats.kind)
                << "\",\"read_requests\":" << geometry.offload_pages
                << ",\"write_requests\":" << geometry.offload_pages
                << ",\"read_bytes\":" << geometry.offload_bytes
                << ",\"write_bytes\":" << geometry.offload_bytes
                << ",\"media_channels\":"
                << sample.external_backing_stats.media_channels
                << ",\"active_media_channels\":"
                << sample.external_backing_stats.active_media_channels
                << ",\"max_device_outstanding\":"
                << sample.external_backing_stats.max_device_outstanding
                << ",\"m2s_payload_bytes\":" << geometry.offload_bytes
                << ",\"m2s_protocol_bytes\":" << m2s_protocol_bytes
                << ",\"m2s_wire_bytes\":" << m2s_wire_bytes
                << ",\"s2m_payload_bytes\":" << geometry.offload_bytes
                << ",\"s2m_protocol_bytes\":" << s2m_protocol_bytes
                << ",\"s2m_wire_bytes\":" << s2m_wire_bytes
                << ",\"representative_controller_utilization\":"
                << sample.external_backing_stats.controller_utilization()
                << ",\"representative_media_utilization\":"
                << sample.external_backing_stats.media_utilization()
                << ",\"representative_m2s_utilization\":"
                << sample.external_backing_stats.m2s_utilization()
                << ",\"representative_s2m_utilization\":"
                << sample.external_backing_stats.s2m_utilization()
                << "},\n";
        }
        out << "      \"conservation\":\"PASS\"\n"
            << "    }" << (index + 1 == cases.size() ? "\n" : ",\n");
    }
    out << "  ],\n"
        << "  \"layer_lifecycle\":{\"provenance\":\"derived\","
        << "\"formula\":\"one-offload-plus-N-restores\","
        << "\"restore_counts\":[";
    for (std::size_t index = 0;
         index < kLifecycleRestoreCounts.size();
         ++index) {
        out << kLifecycleRestoreCounts[index]
            << (index + 1 == kLifecycleRestoreCounts.size() ? "" : ",");
    }
    out << "],\"points\":[\n";
    for (std::size_t restore_index = 0;
         restore_index < kLifecycleRestoreCounts.size();
         ++restore_index) {
        const auto restores = kLifecycleRestoreCounts[restore_index];
        const ProjectedCase* winner = nullptr;
        double winner_elapsed_ns = std::numeric_limits<double>::infinity();
        out << "    {\"restore_count\":" << restores << ",\"cases\":[";
        for (std::size_t case_index = 0;
             case_index < cases.size();
             ++case_index) {
            const auto& item = cases[case_index];
            const auto total_elapsed_ns =
                item.layer_round_trip.projected_offload_elapsed_ns +
                static_cast<double>(restores) *
                    item.layer_round_trip.projected_restore_elapsed_ns;
            if (total_elapsed_ns < winner_elapsed_ns) {
                winner_elapsed_ns = total_elapsed_ns;
                winner = &item;
            }
            out << "{\"name\":\"" << item.name
                << "\",\"offload_elapsed_ns\":"
                << item.layer_round_trip.projected_offload_elapsed_ns
                << ",\"restore_elapsed_ns\":"
                << item.layer_round_trip.projected_restore_elapsed_ns
                << ",\"total_elapsed_ns\":" << total_elapsed_ns
                << ",\"average_elapsed_ns_per_restore\":"
                << total_elapsed_ns / static_cast<double>(restores)
                << "}"
                << (case_index + 1 == cases.size() ? "" : ",");
        }
        if (winner == nullptr) {
            throw std::runtime_error(
                "layer lifecycle has no candidate winner");
        }
        out << "],\"winner\":\"" << winner->name << "\"}"
            << (restore_index + 1 == kLifecycleRestoreCounts.size() ?
                "\n" :
                ",\n");
    }
    const auto find_case = [&](const std::string& name)
        -> const ProjectedCase& {
        const auto match = std::find_if(
            cases.begin(),
            cases.end(),
            [&](const ProjectedCase& item) {
                return item.name == name;
            });
        if (match == cases.end()) {
            throw std::runtime_error(
                "layer lifecycle case is missing: " + name);
        }
        return *match;
    };
    const auto& hbf_case = find_case("hbf");
    const auto write_break_even = [&](const std::string& external_name) {
        const auto& external_case = find_case(external_name);
        for (const auto restores : kLifecycleRestoreCounts) {
            const auto hbf_elapsed =
                hbf_case.layer_round_trip.projected_offload_elapsed_ns +
                static_cast<double>(restores) *
                    hbf_case.layer_round_trip.projected_restore_elapsed_ns;
            const auto external_elapsed =
                external_case.layer_round_trip.projected_offload_elapsed_ns +
                static_cast<double>(restores) *
                    external_case.layer_round_trip.projected_restore_elapsed_ns;
            if (hbf_elapsed <= external_elapsed) {
                out << restores;
                return;
            }
        }
        out << "null";
    };
    out << "  ],\"break_even\":{\"definition\":"
        << "\"first-discrete-N-where-hbf-total-is-no-greater\","
        << "\"hbf_vs_cxl_memory\":";
    write_break_even("cxl-memory");
    out << ",\"hbf_vs_nvme_ssd\":";
    write_break_even("nvme-ssd");
    out << "}},\n"
        << "  \"limitations\":["
        << "\"external profiles are sensitivity points, not calibrated products\","
        << "\"NVMe SSD FTL, GC, wear, tail latency, and failure consistency are not modeled\","
        << "\"CXL cache coherence and full PCIe/NVMe protocol traffic are not modeled\","
        << "\"target-scale timing is projected only after exact-sample convergence\""
        << "]\n"
        << "}\n";
}

void write_csv(
    const Options& options,
    const Geometry& geometry,
    const std::vector<ProjectedCase>& cases) {
    ensure_parent(options.csv_path);
    std::ofstream out(options.csv_path);
    if (!out) {
        throw std::runtime_error(
            "cannot open CSV output: " + options.csv_path.string());
    }
    out << std::setprecision(17);
    out << "case,backing,hbm_capacity_bytes,total_write_bytes,offload_bytes,"
           "target_batches,offload_elapsed_ns,offload_throughput_GBps,"
           "read_elapsed_ns,read_throughput_GBps,drain_tail_ns,"
           "layer_bytes,layer_offload_elapsed_ns,"
           "layer_restore_to_hbm_elapsed_ns,layer_e2e_elapsed_ns,"
           "layer_round_trip_throughput_GBps,"
           "fill_relative_drift,offload_relative_drift,read_relative_drift,"
           "layer_offload_relative_drift,layer_restore_relative_drift,"
           "conservation\n";
    for (const auto& item : cases) {
        out << item.name << ","
            << hbfsim::physical::hybrid::to_string(item.backing) << ","
            << geometry.hbm_capacity_bytes << ","
            << geometry.total_write_bytes << ","
            << geometry.offload_bytes << ","
            << geometry.target_batches << ","
            << item.projected_offload_elapsed_ns << ","
            << throughput_GBps(
                geometry.offload_bytes,
                item.projected_offload_elapsed_ns)
            << "," << item.projected_read_elapsed_ns
            << "," << throughput_GBps(
                geometry.offload_bytes,
                item.projected_read_elapsed_ns)
            << "," << item.projected_drain_tail_ns
            << "," << geometry.layer_bytes
            << "," << item.layer_round_trip.projected_offload_elapsed_ns
            << "," << item.layer_round_trip.projected_restore_elapsed_ns
            << "," << item.layer_round_trip.projected_e2e_elapsed_ns()
            << "," << throughput_GBps(
                checked_mul(
                    geometry.layer_bytes,
                    2,
                    "CSV layer round-trip traffic bytes"),
                item.layer_round_trip.projected_e2e_elapsed_ns())
            << "," << item.fill_relative_drift
            << "," << item.offload_relative_drift
            << "," << item.read_relative_drift
            << "," << item.layer_round_trip.offload_relative_drift
            << "," << item.layer_round_trip.restore_relative_drift
            << ",PASS\n";
    }
}

void print_table(
    const Geometry& geometry,
    const std::vector<ProjectedCase>& cases) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout
        << "case             offloading_ms  offloading_GB/s"
        << "  readback_ms  readback_GB/s  drain_ms  max_drift\n";
    for (const auto& item : cases) {
        std::cout << std::left << std::setw(17) << item.name
                  << std::right << std::setw(14)
                  << item.projected_offload_elapsed_ns / 1e6
                  << std::setw(18)
                  << throughput_GBps(
                      geometry.offload_bytes,
                      item.projected_offload_elapsed_ns)
                  << std::setw(13)
                  << item.projected_read_elapsed_ns / 1e6
                  << std::setw(16)
                  << throughput_GBps(
                      geometry.offload_bytes,
                      item.projected_read_elapsed_ns)
                  << std::setw(10)
                  << item.projected_drain_tail_ns / 1e6
                  << std::setw(11)
                  << 100.0 * std::max(
                      item.fill_relative_drift,
                      std::max(
                          item.offload_relative_drift,
                          item.read_relative_drift))
                  << "%\n";
    }
    std::cout
        << "\nlayer round-trip (" << geometry.layer_bytes / kMiB
        << " MiB, original HBM slots restored and verified)\n"
        << "case             offloading_ms  restore_to_HBM_ms"
        << "  e2e_ms  round_trip_GB/s\n";
    for (const auto& item : cases) {
        const auto& layer = item.layer_round_trip;
        std::cout << std::left << std::setw(17) << item.name
                  << std::right << std::setw(14)
                  << layer.projected_offload_elapsed_ns / 1e6
                  << std::setw(19)
                  << layer.projected_restore_elapsed_ns / 1e6
                  << std::setw(9)
                  << layer.projected_e2e_elapsed_ns() / 1e6
                  << std::setw(19)
                  << throughput_GBps(
                      checked_mul(
                          geometry.layer_bytes,
                          2,
                          "printed layer round-trip traffic bytes"),
                      layer.projected_e2e_elapsed_ns())
                  << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        const auto geometry = derive_geometry(options);

        std::vector<ProjectedCase> cases;
        cases.push_back(project_case(
            geometry,
            "hbf",
            BackingTier::Hbf,
            cxl_memory_profile()));
        cases.push_back(project_case(
            geometry,
            "cxl-memory",
            BackingTier::External,
            cxl_memory_profile()));
        cases.push_back(project_case(
            geometry,
            "nvme-ssd",
            BackingTier::External,
            nvme_ssd_profile()));

        write_json(
            options,
            geometry,
            cases,
            executable_identity(argv[0]));
        write_csv(options, geometry, cases);
        print_table(geometry, cases);
        std::cout << "HBM capacity: "
                  << static_cast<double>(geometry.hbm_capacity_bytes) / kGiB
                  << " GiB, total writes: "
                  << static_cast<double>(geometry.total_write_bytes) / kGiB
                  << " GiB, offloading/readback: "
                  << static_cast<double>(geometry.offload_bytes) / kGiB
                  << " GiB, layer round-trip: "
                  << static_cast<double>(geometry.layer_bytes) / kGiB
                  << " GiB\n"
                  << "wrote: " << options.json_path << "\n"
                  << "wrote: " << options.csv_path << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "overflow_offload_experiment: " << error.what() << "\n";
        return 1;
    }
}
