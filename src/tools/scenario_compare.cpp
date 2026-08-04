#include "physical/external/external_backing_device.hpp"
#include "physical/hbf/hbf_device.hpp"
#include "physical/hbm/hbm_device.hpp"
#include "physical/hybrid/behavioral_tiering_composition.hpp"
#include "physical/hybrid/composition_common.hpp"
#include "physical/hybrid/direct_composition.hpp"
#include "physical/hybrid/layer_streaming_composition.hpp"
#include "tools/sha256.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef HBFSIM_VERSION
#define HBFSIM_VERSION "unknown"
#endif
#ifndef HBFSIM_GIT_COMMIT
#define HBFSIM_GIT_COMMIT "unknown"
#endif
#ifndef HBFSIM_GIT_DIRTY
#define HBFSIM_GIT_DIRTY 0
#endif
#ifndef HBFSIM_BUILD_TYPE
#define HBFSIM_BUILD_TYPE "unknown"
#endif
#ifndef HBFSIM_COMPILER_ID
#define HBFSIM_COMPILER_ID "unknown"
#endif
#ifndef HBFSIM_COMPILER_VERSION
#define HBFSIM_COMPILER_VERSION "unknown"
#endif

using hbfsim::physical::AddressSpace;
using hbfsim::physical::AddressDomain;
using hbfsim::physical::AddressHeatmapSnapshot;
using hbfsim::physical::Breakdown;
using hbfsim::physical::Op;
using hbfsim::physical::PhysicalCompletion;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;
using hbfsim::physical::TraceMode;
using hbfsim::physical::external::ExternalBackingConfig;
using hbfsim::physical::external::ExternalBackingKind;
using hbfsim::physical::external::ExternalBackingStats;
using hbfsim::physical::hbf::HbfConfig;
using hbfsim::physical::hbf::HbfDevice;
using hbfsim::physical::hbm::HbmConfig;
using hbfsim::physical::hbm::HbmDevice;
using hbfsim::physical::hybrid::BaseDieLinkConfig;
using hbfsim::physical::hybrid::BaseDieLinkStats;
using hbfsim::physical::hybrid::BehavioralAdmissionPolicy;
using hbfsim::physical::hybrid::BehavioralTieringConfig;
using hbfsim::physical::hybrid::BehavioralTieringStats;
using hbfsim::physical::hybrid::CompositionRunResult;
using hbfsim::physical::hybrid::DirectPolicy;
using hbfsim::physical::hybrid::DirectRunKnobs;
using hbfsim::physical::hybrid::MemoryRequest;
using hbfsim::physical::hybrid::LayerStreamingComposition;
using hbfsim::physical::hybrid::LayerStreamingConfig;
using hbfsim::physical::hybrid::LayerResidencyContract;
using hbfsim::physical::hybrid::BackingTier;
using hbfsim::physical::hybrid::LayerStreamingStats;
using hbfsim::physical::hybrid::SemanticKind;
using hbfsim::physical::hybrid::SequentialReadWorkload;
using hbfsim::physical::hybrid::all_hbf_policy;
using hbfsim::physical::hybrid::all_hbm_policy;
using hbfsim::physical::hybrid::flat_policy;
using hbfsim::tools::sha256_file;
using hbfsim::tools::sha256_text;
using hbfsim::physical::hybrid::read_only_direct_policy;
using hbfsim::physical::hybrid::run_direct_composition;
using hbfsim::physical::hybrid::run_behavioral_tiering_composition;

namespace {

struct TraceOp {
    std::uint64_t addr = 0;
    Op op = Op::Read;
    std::uint64_t bytes = 64;
    double arrival_ns = 0.0;
    std::size_t line_no = 0;
    SemanticKind kind = SemanticKind::Unknown;
    std::string label;
    std::optional<std::uint64_t> phase;
    std::optional<std::uint64_t> layer;
    std::optional<double> compute_ns;
};

struct Options {
    std::optional<std::string> trace_path;
    // Publication overlays bind the exact trace bytes they were derived
    // from. CLI trace overrides remain possible only when content-identical.
    std::optional<std::string> expected_trace_sha256;
    std::optional<std::uint64_t> expected_trace_file_bytes;
    // Optional full population description for prefix/scaling runs. It is
    // never executed; only initial read-before-write pages seed HBF.
    std::optional<std::string> initial_image_trace_path;
    // File-free fixed-size sequential reads. This is mutually exclusive with
    // --trace and expands lazily into exactly the same MemoryRequest stream.
    std::optional<std::uint64_t> synthetic_sequential_read_bytes;
    std::uint64_t synthetic_sequential_read_base = 0;
    std::optional<std::string> generate_llm_path;
    std::optional<std::string> generate_semantic_llm_path;
    std::optional<std::string> chrome_trace_path;
    std::optional<std::string> summary_csv_path;
    std::optional<std::string> summary_json_path;
    std::optional<std::string> config_out_path;
    bool trace_census_only = false;
    std::uint64_t line_size = 64;
    double interarrival_ns = 1.0;
    std::size_t max_ops = 0;
    std::uint64_t hbm_capacity_bytes = 128ull * 1024ull * 1024ull * 1024ull;
    std::uint64_t flat_hbm_bytes = 64ull * 1024ull * 1024ull;
    bool hbm_capacity_explicit = false;
    bool flat_hbm_bytes_explicit = false;
    std::optional<std::uint64_t> hbf_capacity_bytes_target;
    std::optional<double> hbf_capacity_ratio;
    // Maximum capacity of one ping-pong layer buffer. Runtime sizing uses
    // the actual maximum per-layer data footprint and may allocate less.
    std::uint64_t layer_buffer_bytes = 32ull * 1024ull * 1024ull * 1024ull;
    // Complete Frontier/object-map population. When enabled, capacity and
    // hot/cold placement come from these independently exported values; the
    // selected trace window contributes traffic only.
    bool explicit_residency_contract = false;
    std::optional<std::uint64_t> residency_page_size_bytes;
    std::optional<std::uint64_t> residency_unique_footprint_bytes;
    std::optional<std::uint64_t> residency_immutable_weight_bytes;
    std::optional<std::uint64_t> residency_immutable_weight_pages;
    std::optional<std::uint64_t> residency_static_weight_pages;
    std::optional<std::uint64_t> residency_runtime_overhead_bytes;
    std::optional<std::uint64_t> residency_block_table_bytes;
    std::optional<std::uint64_t> residency_active_buffer_bytes;
    std::optional<std::uint64_t> residency_kv_region_begin;
    std::optional<std::uint64_t> residency_kv_block_stride_bytes;
    std::optional<std::uint64_t> residency_logical_kv_blocks;
    std::optional<std::uint64_t> residency_hot_kv_blocks;
    // Online behavior-only HBM residency. Zero owns the complete configured
    // HBM capacity. The bounded history is metadata, not data capacity.
    std::uint64_t behavioral_hbm_bytes = 0;
    std::uint32_t behavioral_promotion_threshold = 2;
    std::uint64_t behavioral_history_pages = 1'048'576;
    // Completion-order page-transaction credits. Layer streaming applies the
    // same numeric bound independently to foreground HBM and backing traffic.
    std::size_t max_outstanding_requests = 0;  // 0 = open loop
    std::size_t max_hbm_outstanding_requests = 0;
    std::size_t max_hbf_outstanding_requests = 0;
    // Fixed-size spatial observability. This is independent of trace mode and
    // bounded by configuration rather than operation count.
    std::size_t address_heatmap_bins = hbfsim::physical::kDefaultAddressHeatmapBins;
    // Exploratory PER-STACK D2D link (every HBF stack has its own interface).
    double base_die_link_read_bw_GBps = 512.0;
    double base_die_link_write_bw_GBps = 128.0;
    double base_die_link_latency_ns = 20.0;
    // No-HBF backing profile. Optional numeric overrides are applied after
    // resolving the selected CXL-memory or NVMe-SSD envelope.
    ExternalBackingKind external_backing_kind = ExternalBackingKind::CxlMemory;
    std::optional<std::uint64_t> external_backing_capacity_bytes;
    std::optional<std::uint64_t> external_backing_page_size;
    std::optional<std::uint32_t> external_backing_media_channels;
    std::optional<std::uint32_t>
        external_backing_max_outstanding_requests;
    std::optional<double> external_backing_controller_issue_ns;
    std::optional<double> external_backing_controller_processing_ns;
    std::optional<double> external_backing_media_read_latency_ns;
    std::optional<double> external_backing_media_write_latency_ns;
    std::optional<double> external_backing_media_read_bw_GBps;
    std::optional<double> external_backing_media_write_bw_GBps;
    std::optional<double> external_backing_m2s_bw_GBps;
    std::optional<double> external_backing_s2m_bw_GBps;
    std::optional<double> external_backing_one_way_propagation_ns;
    std::optional<std::uint32_t> external_backing_command_bytes;
    std::optional<std::uint32_t> external_backing_completion_bytes;
    std::size_t llm_tokens = 32;
    std::size_t llm_layers = 8;
    std::uint64_t llm_weight_base = 0x1000'0000ull;
    std::uint64_t llm_kv_base = 0x4000'0000ull;
    std::uint64_t llm_scratch_base = 0x7000'0000ull;
    std::uint32_t hbm_stacks = 1;
    std::uint32_t hbm_channels = 4;
    std::uint32_t hbm_pseudo_channels = 2;
    std::uint32_t hbm_bank_groups_per_pseudo_channel = 4;
    std::uint32_t hbm_banks_per_group = 4;
    std::uint64_t hbm_channel_row_size_bytes = 2048;
    std::uint32_t hbm_channel_width_bits = 64;
    std::uint32_t hbm_burst_length = 8;
    double hbm_pin_rate_Gbps = 6.4;
    std::uint32_t hbm_data_rate_per_command_clock = 4;
    double hbm_address_mapping_ns = 0.0;
    double hbm_trcdrd_ns = 14.0;
    double hbm_trcdwr_ns = 14.0;
    double hbm_tcl_ns = 14.0;
    double hbm_tcwl_ns = 10.0;
    double hbm_trp_ns = 14.0;
    double hbm_tras_ns = 32.0;
    double hbm_trc_ns = 46.0;
    double hbm_twr_ns = 15.0;
    double hbm_trtp_ns = 7.5;
    std::uint32_t hbm_tccd_s_cycles = 2;
    std::uint32_t hbm_tccd_l_cycles = 4;
    double hbm_trrd_s_ns = 4.0;
    double hbm_trrd_l_ns = 6.0;
    double hbm_tfaw_ns = 20.0;
    double hbm_twtr_s_ns = 4.0;
    double hbm_twtr_l_ns = 8.0;
    double hbm_trtw_ns = 8.0;
    bool hbm_refresh_enabled = false;
    bool hbm_same_bank_refresh = false;
    double hbm_trefi_ns = 3900.0;
    double hbm_trfc_ns = 350.0;
    double hbm_trfcsb_ns = 160.0;
    std::uint32_t hbm_queue_depth = 32;
    double hbm_frfcfs_cap_ns = 5000.0;
    std::uint32_t hbf_stacks = 4;
    std::uint32_t hbf_channels = 4;
    std::uint32_t hbf_dies_per_channel = 2;
    std::uint32_t hbf_planes_per_die = 4;
    std::uint32_t hbf_blocks_per_plane = 2048;
    std::uint32_t hbf_pages_per_block = 256;
    std::uint64_t hbf_page_size = 2048;
    std::uint64_t hbf_oob_bytes = 128;
    std::uint32_t hbf_media_lanes_per_plane = 16;
    std::uint32_t hbf_subarrays_per_plane = 0;
    std::uint32_t hbf_page_buffer_banks_per_plane = 1;
    double hbf_read_ns = 4000.0;
    double hbf_program_ns = 75000.0;
    double hbf_program_verify_ns = 5000.0;
    double hbf_erase_ns = 2'000'000.0;
    double hbf_ecc_decode_latency_ns = 500.0;
    double hbf_ecc_encode_latency_ns = 500.0;
    // Default geometry has eight dies per stack and 2048 B + 128 B
    // codewords: 136 raw GB/s/die provisions 1024 payload GB/s/stack.
    double hbf_ecc_decode_raw_bw_GBps_per_die = 136.0;
    double hbf_ecc_encode_raw_bw_GBps_per_die = 136.0;
    // Internal channel data and ECC rates include OOB. The shared TSV also
    // serializes internal commands; HBIO and logic SRAM carry decoded payload.
    // Defaults are library fallbacks; published profiles bind all rates.
    double hbf_channel_bw_GBps = 272.0;
    double hbf_hbio_bw_GBps = 1024.0;
    double hbf_tsv_bw_GBps = 1120.0;
    double hbf_media_lane_bw_GBps = 2048.0;
    double hbf_logic_sram_bw_GBps = 2048.0;
    double hbf_page_buffer_bw_GBps = 2048.0;
    std::uint64_t hbf_ctrl_dram_bytes = 0;  // 0 = exact full-resident L2P size
    double hbf_ctrl_dram_latency_ns = 100.0;
    double hbf_ctrl_dram_issue_ns = 1.0;
    double hbf_flash_tsu_issue_ns = 10.0;
    double hbf_logic_scheduler_issue_ns = 2.0;
    bool hbf_batch_activation = true;
    std::uint64_t hbf_read_buffer_pages = 1024;
    std::uint64_t hbf_page_read_queue_depth_per_stack = 4096;
    std::uint64_t static_direct_hbm_bytes = 0;
    // Empty = run all scenarios. A per-case selection lets a config be used
    // at capacities its auxiliary scenarios could not survive.
    std::vector<std::string> scenarios;
    std::uint64_t hbf_gc_low_watermark_pages = 4096;
    std::uint64_t hbf_gc_hard_watermark_pages = 0;
    std::uint64_t hbf_gc_reserved_free_blocks_per_plane = 0;
    double hbf_gc_wear_leveling_weight = 0.0;
    bool hbf_write_coalescing = false;
    bool hbf_write_buffer_completion_requires_flush = false;
    std::uint64_t hbf_write_buffer_pages = 1024;
    std::uint64_t hbf_write_buffer_flush_threshold_pages = 0;
    // Cooperative HBM/HBF write staging region (0 = off): HBF-bound writes
    // in the direct compositions land in HBM first and destage to the FTL
    // in the background.
    std::uint64_t hbf_hbm_write_buffer_bytes = 0;
    // "deferred" streams the region out after user traffic quiets;
    // "streamed" starts chains with production for layer-structured traces.
    std::string hbf_hbm_write_buffer_destage = "deferred";
    TraceMode trace_mode = TraceMode::Off;
};

constexpr std::uint64_t KiB = 1024ull;
constexpr std::uint64_t MiB = KiB * 1024ull;
constexpr std::uint64_t GiB = MiB * 1024ull;

struct RunProvenance {
    std::vector<std::string> command;
    std::string working_directory;
    std::string executable_path;
    std::string executable_sha256;
    std::uint64_t executable_file_bytes = 0;
    std::string trace_sha256;
    std::uint64_t trace_file_bytes = 0;
    std::string initial_image_trace_sha256;
    std::uint64_t initial_image_trace_file_bytes = 0;
};

struct LatencyDistribution {
    std::uint64_t count = 0;
    double sum_work_ns = 0.0;
    double average_ns = 0.0;
    double p50_ns = 0.0;
    double p95_ns = 0.0;
    double max_ns = 0.0;
};

struct ScenarioResult {
    std::string name;
    std::vector<double> service_latencies_ns;
    std::vector<double> offered_latencies_ns;
    std::vector<double> source_latencies_ns;
    std::optional<LatencyDistribution> service_latency_distribution;
    std::optional<LatencyDistribution> offered_latency_distribution;
    std::optional<LatencyDistribution> source_latency_distribution;
    std::vector<PhysicalCompletion> completions;
    std::uint64_t ops = 0;
    std::uint64_t reads = 0;
    std::uint64_t writes = 0;
    std::uint64_t logical_bytes = 0;
    std::uint64_t hbm_accesses = 0;
    std::uint64_t hbf_accesses = 0;
    std::uint64_t external_accesses = 0;
    std::uint64_t hbm_user_accesses = 0;
    std::uint64_t hbf_user_accesses = 0;
    std::uint64_t hbf_direct_user_ops = 0;
    std::uint64_t hbm_background_accesses = 0;
    std::uint64_t hbf_background_accesses = 0;
    std::uint64_t external_background_accesses = 0;
    std::optional<LayerStreamingStats> layer_streaming_stats;
    std::optional<BehavioralTieringStats> behavioral_tiering_stats;
    std::uint64_t background_hbf_writes = 0;
    std::uint64_t hbf_static_read_bytes = 0;
    std::uint64_t hbf_logical_read_bytes = 0;
    std::uint64_t hbf_backing_write_bytes = 0;
    std::uint64_t external_backing_read_bytes = 0;
    std::uint64_t external_backing_write_bytes = 0;
    std::uint64_t hbm_foreground_bytes = 0;
    std::uint64_t hbm_streaming_write_bytes = 0;
    std::uint64_t hbm_write_buffer_user_write_bytes = 0;
    std::uint64_t hbm_write_buffer_destaged_bytes = 0;
    std::uint64_t hbm_write_buffer_peak_bytes = 0;
    std::uint64_t hbm_write_buffer_full_waits = 0;
    double hbm_write_buffer_wait_ns = 0.0;
    bool cooperative_write_controller_present = false;
    BaseDieLinkStats base_die_link_stats;
    std::uint64_t phase_barriers = 0;
    std::uint64_t phase_dependency_waited_ops = 0;
    double phase_dependency_wait_work_ns = 0.0;
    double phase_dependency_max_wait_ns = 0.0;
    std::uint64_t front_end_admission_waited_ops = 0;
    double front_end_admission_wait_work_ns = 0.0;
    double front_end_admission_max_wait_ns = 0.0;
    double finish_ns = 0.0;
    // Absolute completion of the last USER op. finish_ns additionally covers
    // write-buffer/mapping drain and late background fills. Reporting derives
    // both elapsed spans from first_arrival_ns and never divides by an
    // absolute timestamp.
    double user_finish_ns = 0.0;
    // Trace time origin and arrival extent. Absolute completion timestamps
    // remain useful for event correlation, while throughput is calculated
    // only from elapsed intervals derived from this origin.
    double first_arrival_ns = 0.0;
    double last_arrival_ns = 0.0;
    bool owns_arrival_frontier = false;
    bool has_hbm = false;
    bool has_hbf = false;
    bool has_external_backing = false;
    hbfsim::physical::hbm::HbmStats hbm_stats;
    hbfsim::physical::hbf::HbfStats hbf_stats;
    ExternalBackingStats external_backing_stats;
    std::optional<AddressHeatmapSnapshot> address_heatmap;
    std::vector<std::string> warnings;
};

constexpr const char* kAllHbmScenario = "all-HBM";
constexpr const char* kAllHbfScenario = "all-HBF";
constexpr const char* kFlatScenario = "HBM-HBF-Flat";
constexpr const char* kStaticDirectScenario = "HBF-static-direct-read";
constexpr const char* kLayerStreamingScenario = "HBM+HBF-layer-streaming";
constexpr const char* kExternalLayerStreamingScenario =
    "HBM+External-layer-streaming";
constexpr const char* kDemandFillScenario = "HBM+HBF-demand-fill";
constexpr const char* kBehavioralTieringScenario =
    "HBM+HBF-behavioral-placement";

bool scenario_selected(const Options& options, std::string_view name) {
    return options.scenarios.empty() || std::any_of(
        options.scenarios.begin(),
        options.scenarios.end(),
        [name](const std::string& candidate) { return candidate == name; });
}

std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string upper(std::string value) {
    for (auto& ch : value) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool parse_op_token(const std::string& token, Op& op) {
    const auto t = upper(token);
    if (t == "R" || t == "READ" || t == "LD" || t == "LOAD") {
        op = Op::Read;
        return true;
    }
    if (t == "W" || t == "WRITE" || t == "ST" || t == "STORE") {
        op = Op::Write;
        return true;
    }
    return false;
}

std::string normalize_token(std::string value) {
    value = trim(std::move(value));
    for (auto& ch : value) {
        if (ch == '-' || ch == '.') {
            ch = '_';
        } else {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
    }
    return value;
}

std::optional<SemanticKind> parse_semantic_kind_token(std::string token) {
    const auto eq = token.find('=');
    if (eq != std::string::npos) {
        token = token.substr(eq + 1);
    }
    const auto value = normalize_token(std::move(token));
    if (value.empty() || value == "unknown" || value == "none") {
        return SemanticKind::Unknown;
    }
    if (value == "model" || value == "weights" || value == "model_weight" ||
        value == "model_weights" || value == "weight") {
        return SemanticKind::ModelWeights;
    }
    if (value == "shared" || value == "shared_context" || value == "shared_kv" ||
        value == "prompt_context" || value == "prefill_context") {
        return SemanticKind::SharedContext;
    }
    if (value == "generated" || value == "generated_context" || value == "generated_kv" ||
        value == "decode_context" || value == "kv_cache") {
        return SemanticKind::GeneratedContext;
    }
    if (value == "scratch" || value == "activation" || value == "activations" ||
        value == "temp" || value == "temporary") {
        return SemanticKind::Scratch;
    }
    if (value == "metadata" || value == "meta" || value == "page_table" ||
        value == "mapping") {
        return SemanticKind::Metadata;
    }
    return std::nullopt;
}

int integer_base(const std::string& token) {
    // Base 10 unless explicitly hex: stoull's auto-detected base would read
    // zero-padded decimal tokens as octal.
    if (token.size() > 2 && token[0] == '0' && (token[1] == 'x' || token[1] == 'X')) {
        return 16;
    }
    return 10;
}

bool try_parse_u64(const std::string& token, std::uint64_t& out) {
    if (token.empty() || token.front() == '-') {
        return false;
    }
    try {
        std::size_t used = 0;
        const auto value = std::stoull(token, &used, integer_base(token));
        if (used == token.size()) {
            out = value;
            return true;
        }
    } catch (const std::exception&) {
    }
    return false;
}

bool parse_bool(std::string value) {
    value = upper(trim(std::move(value)));
    if (value == "1" || value == "TRUE" || value == "YES" || value == "ON") {
        return true;
    }
    if (value == "0" || value == "FALSE" || value == "NO" || value == "OFF") {
        return false;
    }
    throw std::runtime_error("invalid boolean value: " + value);
}

bool is_lower_hex_sha256(const std::string& value) {
    return value.size() == 64 &&
        std::all_of(
            value.begin(),
            value.end(),
            [](char ch) {
                return (ch >= '0' && ch <= '9') ||
                    (ch >= 'a' && ch <= 'f');
            });
}

double parse_double(const std::string& token, const char* name) {
    try {
        std::size_t used = 0;
        const auto value = std::stod(token, &used);
        if (used == token.size() && std::isfinite(value)) {
            return value;
        }
    } catch (const std::exception&) {
    }
    throw std::runtime_error(std::string(name) + ": invalid number: " + token);
}

std::uint64_t parse_u64(const std::string& token) {
    if (token.empty() || token.front() == '-') {
        throw std::runtime_error("invalid non-negative integer token: " + token);
    }
    std::size_t used = 0;
    std::uint64_t value = 0;
    try {
        value = std::stoull(token, &used, integer_base(token));
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer token: " + token);
    }
    if (used != token.size()) {
        throw std::runtime_error("invalid integer token: " + token);
    }
    return value;
}

std::uint32_t parse_u32(const std::string& token, const char* name) {
    const auto value = parse_u64(token);
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds uint32_t range");
    }
    return static_cast<std::uint32_t>(value);
}

std::size_t parse_size(const std::string& token, const char* name) {
    const auto value = parse_u64(token);
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds size_t range");
    }
    return static_cast<std::size_t>(value);
}

std::uint64_t checked_mul_u64(std::uint64_t lhs, std::uint64_t rhs, const std::string& name) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(name + " overflows uint64_t");
    }
    return lhs * rhs;
}

std::uint64_t checked_add_u64(std::uint64_t lhs, std::uint64_t rhs, const std::string& name) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(name + " overflows uint64_t");
    }
    return lhs + rhs;
}

std::uint64_t ceil_div_u64(std::uint64_t value, std::uint64_t divisor) {
    if (divisor == 0) {
        throw std::runtime_error("internal error: divide by zero");
    }
    return value == 0 ? 0 :
        checked_add_u64(value - 1, divisor, "ceil division") / divisor;
}

SequentialReadWorkload make_synthetic_sequential_workload(
    const Options& options) {
    if (!options.synthetic_sequential_read_bytes) {
        throw std::runtime_error("synthetic sequential workload is not configured");
    }
    const auto request_count =
        *options.synthetic_sequential_read_bytes / options.line_size;
    if (request_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "synthetic sequential request count exceeds size_t range");
    }
    return SequentialReadWorkload{
        .base_addr = options.synthetic_sequential_read_base,
        .request_bytes = options.line_size,
        .request_count = static_cast<std::size_t>(request_count),
        .first_arrival_ns = 0.0,
        .interarrival_ns = options.interarrival_ns,
    };
}

std::string synthetic_sequential_descriptor(
    const Options& options,
    const SequentialReadWorkload& workload) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "hbfsim.synthetic-sequential-read.v1\n"
        << "base_addr=" << workload.base_addr << '\n'
        << "request_bytes=" << workload.request_bytes << '\n'
        << "request_count=" << workload.request_count << '\n'
        << "total_bytes=" << *options.synthetic_sequential_read_bytes << '\n'
        << "first_arrival_ns=" << workload.first_arrival_ns << '\n'
        << "interarrival_ns=" << workload.interarrival_ns << '\n';
    return out.str();
}

std::string format_bytes(std::uint64_t bytes) {
    std::ostringstream out;
    out << bytes << "B";
    if (bytes % GiB == 0) {
        out << " (" << (bytes / GiB) << "GiB)";
    } else if (bytes % MiB == 0) {
        out << " (" << (bytes / MiB) << "MiB)";
    } else if (bytes % KiB == 0) {
        out << " (" << (bytes / KiB) << "KiB)";
    }
    return out.str();
}

TraceMode parse_trace_mode_value(const std::string& value);

std::optional<TraceOp> parse_trace_line(
    std::string line,
    std::size_t line_no,
    std::uint64_t line_size,
    double arrival_ns,
    std::set<std::string>& unknown_labels) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) {
        line = line.substr(0, hash);
    }
    line = trim(line);
    if (line.empty()) {
        return std::nullopt;
    }

    std::istringstream in(line);
    std::vector<std::string> tokens;
    for (std::string token; in >> token;) {
        tokens.push_back(token);
    }
    if (tokens.size() < 2) {
        throw std::runtime_error("trace line " + std::to_string(line_no) +
            " must be Ramulator-compatible: <addr> <R|W>");
    }

    Op op = Op::Read;
    std::uint64_t addr = 0;
    std::size_t op_index = 0;
    std::size_t addr_index = 0;
    if (parse_op_token(tokens[1], op)) {
        addr = parse_u64(tokens[0]);
        addr_index = 0;
        op_index = 1;
    } else if (parse_op_token(tokens[0], op)) {
        addr = parse_u64(tokens[1]);
        addr_index = 1;
        op_index = 0;
    } else {
        throw std::runtime_error("trace line " + std::to_string(line_no) +
            " has no R/W operation token");
    }

    auto bytes = line_size;
    auto kind = SemanticKind::Unknown;
    auto arrival = arrival_ns;
    std::string label;
    std::optional<std::uint64_t> phase;
    std::optional<std::uint64_t> layer;
    std::optional<double> compute_ns;
    bool saw_phase = false;
    bool saw_layer = false;
    bool saw_compute = false;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i == addr_index || i == op_index) {
            continue;
        }
        auto token = tokens[i];
        const auto eq = token.find('=');
        const auto key = eq == std::string::npos ? std::string{} :
            normalize_token(token.substr(0, eq));
        const auto value = eq == std::string::npos ? token : token.substr(eq + 1);
        std::uint64_t parsed_bytes = 0;
        if ((key.empty() || key == "bytes" || key == "size") &&
            try_parse_u64(value, parsed_bytes)) {
            bytes = parsed_bytes;
            continue;
        }
        if (key == "at" || key == "arrival") {
            // Trace-provided arrival time in ns (e.g. from ASTRA-sim 3.0
            // memory-channel export). Overrides the synthesized
            // interarrival-ns arrival for this op only.
            bool ok = false;
            try {
                std::size_t used = 0;
                const double parsed = std::stod(value, &used);
                if (used == value.size() && std::isfinite(parsed) && parsed >= 0.0) {
                    arrival = parsed;
                    ok = true;
                }
            } catch (const std::exception&) {
            }
            if (!ok) {
                throw std::runtime_error("trace line " + std::to_string(line_no) +
                    " has invalid arrival time: " + value);
            }
            continue;
        }
        if (key == "phase") {
            if (saw_phase) {
                throw std::runtime_error(
                    "trace line " + std::to_string(line_no) +
                    " has duplicate phase field");
            }
            std::uint64_t parsed_phase = 0;
            if (!try_parse_u64(value, parsed_phase)) {
                throw std::runtime_error("trace line " + std::to_string(line_no) +
                    " has invalid phase id: " + value);
            }
            phase = parsed_phase;
            saw_phase = true;
            continue;
        }
        if (key == "layer") {
            if (saw_layer) {
                throw std::runtime_error(
                    "trace line " + std::to_string(line_no) +
                    " has duplicate layer field");
            }
            std::uint64_t parsed_layer = 0;
            if (!try_parse_u64(value, parsed_layer)) {
                throw std::runtime_error("trace line " + std::to_string(line_no) +
                    " has invalid layer id: " + value);
            }
            layer = parsed_layer;
            saw_layer = true;
            continue;
        }
        if (key == "compute_ns" || key == "compute") {
            if (saw_compute) {
                throw std::runtime_error(
                    "trace line " + std::to_string(line_no) +
                    " has duplicate compute_ns field");
            }
            bool ok = false;
            try {
                std::size_t used = 0;
                const double parsed = std::stod(value, &used);
                if (used == value.size() && std::isfinite(parsed) &&
                    parsed >= 0.0) {
                    compute_ns = parsed;
                    ok = true;
                }
            } catch (const std::exception&) {
            }
            if (!ok) {
                throw std::runtime_error(
                    "trace line " + std::to_string(line_no) +
                    " has invalid compute_ns: " + value);
            }
            saw_compute = true;
            continue;
        }
        if (key == "label" || key == "region" || key == "name") {
            label = value;
            continue;
        }
        if (key.empty() || key == "kind" || key == "semantic" || key == "type") {
            if (const auto parsed_kind = parse_semantic_kind_token(value)) {
                kind = *parsed_kind;
                if (label.empty()) {
                    label = value;
                }
                continue;
            }
            if (!key.empty()) {
                // An explicit kind= that does not parse is a typo, and silently
                // treating it as a label would change routing in the semantic
                // scenarios.
                throw std::runtime_error("trace line " + std::to_string(line_no) +
                    " has unknown semantic kind: " + value);
            }
        }
        if (label.empty()) {
            label = token;
            unknown_labels.insert(token);
        }
    }
    if (bytes == 0) {
        throw std::runtime_error("trace line " + std::to_string(line_no) +
            " has zero-byte memory operation");
    }
    if (bytes - 1 > std::numeric_limits<std::uint64_t>::max() - addr) {
        throw std::runtime_error("trace line " + std::to_string(line_no) +
            " address range overflows uint64_t");
    }

    return TraceOp{
        .addr = addr,
        .op = op,
        .bytes = bytes,
        .arrival_ns = arrival,
        .line_no = line_no,
        .kind = kind,
        .label = std::move(label),
        .phase = phase,
        .layer = layer,
        .compute_ns = compute_ns,
    };
}

void apply_option(Options& options, const std::string& raw_key, const std::optional<std::string>& value) {
    const auto key = raw_key.starts_with("--") ? raw_key.substr(2) : raw_key;
    auto need = [&]() -> const std::string& {
        if (!value) {
            throw std::runtime_error("option requires a value: " + raw_key);
        }
        return *value;
    };

    if (key == "trace") {
        options.trace_path = need();
    } else if (key == "expected-trace-sha256") {
        options.expected_trace_sha256 = need();
    } else if (key == "expected-trace-bytes") {
        options.expected_trace_file_bytes = parse_u64(need());
    } else if (key == "initial-image-trace") {
        options.initial_image_trace_path = need();
    } else if (key == "synthetic-sequential-read-bytes") {
        options.synthetic_sequential_read_bytes = parse_u64(need());
    } else if (key == "synthetic-sequential-read-base") {
        options.synthetic_sequential_read_base = parse_u64(need());
    } else if (key == "generate-llm") {
        options.generate_llm_path = need();
    } else if (key == "generate-semantic-llm") {
        options.generate_semantic_llm_path = need();
    } else if (key == "chrome-trace") {
        options.chrome_trace_path = need();
    } else if (key == "summary-csv") {
        options.summary_csv_path = need();
    } else if (key == "summary-json") {
        options.summary_json_path = need();
    } else if (key == "config-out") {
        options.config_out_path = need();
    } else if (key == "trace-census-only") {
        options.trace_census_only = parse_bool(need());
    } else if (key == "line-size") {
        options.line_size = parse_u64(need());
    } else if (key == "interarrival-ns") {
        options.interarrival_ns = parse_double(need(), "--interarrival-ns");
    } else if (key == "max-ops") {
        options.max_ops = parse_size(need(), "--max-ops");
    } else if (key == "hbm-capacity-bytes") {
        options.hbm_capacity_bytes = parse_u64(need());
        options.hbm_capacity_explicit = true;
    } else if (key == "flat-hbm-bytes") {
        options.flat_hbm_bytes = parse_u64(need());
        options.flat_hbm_bytes_explicit = true;
    } else if (key == "hbf-capacity-bytes") {
        options.hbf_capacity_bytes_target = parse_u64(need());
    } else if (key == "hbf-capacity-ratio") {
        options.hbf_capacity_ratio = parse_double(need(), "--hbf-capacity-ratio");
    } else if (key == "layer-buffer-bytes") {
        options.layer_buffer_bytes = parse_u64(need());
    } else if (key == "explicit-residency-contract") {
        options.explicit_residency_contract = parse_bool(need());
    } else if (key == "residency-page-size-bytes") {
        options.residency_page_size_bytes = parse_u64(need());
    } else if (key == "residency-unique-footprint-bytes") {
        options.residency_unique_footprint_bytes = parse_u64(need());
    } else if (key == "residency-immutable-weight-bytes") {
        options.residency_immutable_weight_bytes = parse_u64(need());
    } else if (key == "residency-immutable-weight-pages") {
        options.residency_immutable_weight_pages = parse_u64(need());
    } else if (key == "residency-static-weight-pages") {
        options.residency_static_weight_pages = parse_u64(need());
    } else if (key == "residency-runtime-overhead-bytes") {
        options.residency_runtime_overhead_bytes = parse_u64(need());
    } else if (key == "residency-block-table-bytes") {
        options.residency_block_table_bytes = parse_u64(need());
    } else if (key == "residency-active-buffer-bytes") {
        options.residency_active_buffer_bytes = parse_u64(need());
    } else if (key == "residency-kv-region-begin") {
        options.residency_kv_region_begin = parse_u64(need());
    } else if (key == "residency-kv-block-stride-bytes") {
        options.residency_kv_block_stride_bytes = parse_u64(need());
    } else if (key == "residency-logical-kv-blocks") {
        options.residency_logical_kv_blocks = parse_u64(need());
    } else if (key == "residency-hot-kv-blocks") {
        options.residency_hot_kv_blocks = parse_u64(need());
    } else if (key == "behavioral-hbm-bytes") {
        options.behavioral_hbm_bytes = parse_u64(need());
    } else if (key == "behavioral-promotion-threshold") {
        options.behavioral_promotion_threshold =
            parse_u32(need(), "--behavioral-promotion-threshold");
    } else if (key == "behavioral-history-pages") {
        options.behavioral_history_pages = parse_u64(need());
    } else if (key == "max-outstanding-requests") {
        options.max_outstanding_requests =
            parse_size(need(), "--max-outstanding-requests");
    } else if (key == "max-hbm-outstanding-requests") {
        options.max_hbm_outstanding_requests =
            parse_size(need(), "--max-hbm-outstanding-requests");
    } else if (key == "max-hbf-outstanding-requests") {
        options.max_hbf_outstanding_requests =
            parse_size(need(), "--max-hbf-outstanding-requests");
    } else if (key == "address-heatmap-bins") {
        options.address_heatmap_bins =
            parse_size(need(), "--address-heatmap-bins");
    } else if (key == "base-die-link-read-bw") {
        options.base_die_link_read_bw_GBps = parse_double(need(), "--base-die-link-read-bw");
    } else if (key == "base-die-link-write-bw") {
        options.base_die_link_write_bw_GBps = parse_double(need(), "--base-die-link-write-bw");
    } else if (key == "base-die-link-latency-ns") {
        options.base_die_link_latency_ns = parse_double(need(), "--base-die-link-latency-ns");
    } else if (key == "external-backing-kind") {
        options.external_backing_kind =
            hbfsim::physical::external::parse_external_backing_kind(need());
    } else if (key == "external-backing-capacity-bytes") {
        options.external_backing_capacity_bytes = parse_u64(need());
    } else if (key == "external-backing-page-size") {
        options.external_backing_page_size = parse_u64(need());
    } else if (key == "external-backing-media-channels") {
        options.external_backing_media_channels =
            parse_u32(need(), "--external-backing-media-channels");
    } else if (key == "external-backing-max-outstanding-requests") {
        options.external_backing_max_outstanding_requests =
            parse_u32(
                need(),
                "--external-backing-max-outstanding-requests");
    } else if (key == "external-backing-controller-issue-ns") {
        options.external_backing_controller_issue_ns =
            parse_double(
                need(), "--external-backing-controller-issue-ns");
    } else if (key == "external-backing-controller-processing-ns") {
        options.external_backing_controller_processing_ns =
            parse_double(
                need(),
                "--external-backing-controller-processing-ns");
    } else if (key == "external-backing-media-read-latency-ns") {
        options.external_backing_media_read_latency_ns =
            parse_double(
                need(), "--external-backing-media-read-latency-ns");
    } else if (key == "external-backing-media-write-latency-ns") {
        options.external_backing_media_write_latency_ns =
            parse_double(
                need(), "--external-backing-media-write-latency-ns");
    } else if (key == "external-backing-media-read-bw") {
        options.external_backing_media_read_bw_GBps =
            parse_double(need(), "--external-backing-media-read-bw");
    } else if (key == "external-backing-media-write-bw") {
        options.external_backing_media_write_bw_GBps =
            parse_double(need(), "--external-backing-media-write-bw");
    } else if (key == "external-backing-m2s-bw") {
        options.external_backing_m2s_bw_GBps =
            parse_double(need(), "--external-backing-m2s-bw");
    } else if (key == "external-backing-s2m-bw") {
        options.external_backing_s2m_bw_GBps =
            parse_double(need(), "--external-backing-s2m-bw");
    } else if (key == "external-backing-one-way-propagation-ns") {
        options.external_backing_one_way_propagation_ns =
            parse_double(
                need(),
                "--external-backing-one-way-propagation-ns");
    } else if (key == "external-backing-command-bytes") {
        options.external_backing_command_bytes =
            parse_u32(need(), "--external-backing-command-bytes");
    } else if (key == "external-backing-completion-bytes") {
        options.external_backing_completion_bytes =
            parse_u32(need(), "--external-backing-completion-bytes");
    } else if (key == "llm-tokens") {
        options.llm_tokens = parse_size(need(), "--llm-tokens");
    } else if (key == "llm-layers") {
        options.llm_layers = parse_size(need(), "--llm-layers");
    } else if (key == "llm-weight-base") {
        options.llm_weight_base = parse_u64(need());
    } else if (key == "llm-kv-base") {
        options.llm_kv_base = parse_u64(need());
    } else if (key == "llm-scratch-base") {
        options.llm_scratch_base = parse_u64(need());
    } else if (key == "hbm-stacks") {
        options.hbm_stacks = parse_u32(need(), "--hbm-stacks");
    } else if (key == "hbm-channels") {
        options.hbm_channels = parse_u32(need(), "--hbm-channels");
    } else if (key == "hbm-pseudo-channels") {
        options.hbm_pseudo_channels = parse_u32(need(), "--hbm-pseudo-channels");
    } else if (key == "hbm-bank-groups-per-pseudo-channel") {
        options.hbm_bank_groups_per_pseudo_channel =
            parse_u32(need(), "--hbm-bank-groups-per-pseudo-channel");
    } else if (key == "hbm-banks-per-group") {
        options.hbm_banks_per_group = parse_u32(need(), "--hbm-banks-per-group");
    } else if (key == "hbm-channel-row-size-bytes") {
        options.hbm_channel_row_size_bytes = parse_u64(need());
    } else if (key == "hbm-channel-width-bits") {
        options.hbm_channel_width_bits = parse_u32(need(), "--hbm-channel-width-bits");
    } else if (key == "hbm-burst-length") {
        options.hbm_burst_length = parse_u32(need(), "--hbm-burst-length");
    } else if (key == "hbm-pin-rate-gbps") {
        options.hbm_pin_rate_Gbps = parse_double(need(), "--hbm-pin-rate-gbps");
    } else if (key == "hbm-data-rate-per-command-clock") {
        options.hbm_data_rate_per_command_clock =
            parse_u32(need(), "--hbm-data-rate-per-command-clock");
    } else if (key == "hbm-address-mapping-ns") {
        options.hbm_address_mapping_ns =
            parse_double(need(), "--hbm-address-mapping-ns");
    } else if (key == "hbm-trcdrd-ns") {
        options.hbm_trcdrd_ns = parse_double(need(), "--hbm-trcdrd-ns");
    } else if (key == "hbm-trcdwr-ns") {
        options.hbm_trcdwr_ns = parse_double(need(), "--hbm-trcdwr-ns");
    } else if (key == "hbm-tcl-ns") {
        options.hbm_tcl_ns = parse_double(need(), "--hbm-tcl-ns");
    } else if (key == "hbm-tcwl-ns") {
        options.hbm_tcwl_ns = parse_double(need(), "--hbm-tcwl-ns");
    } else if (key == "hbm-trp-ns") {
        options.hbm_trp_ns = parse_double(need(), "--hbm-trp-ns");
    } else if (key == "hbm-tras-ns") {
        options.hbm_tras_ns = parse_double(need(), "--hbm-tras-ns");
    } else if (key == "hbm-trc-ns") {
        options.hbm_trc_ns = parse_double(need(), "--hbm-trc-ns");
    } else if (key == "hbm-twr-ns") {
        options.hbm_twr_ns = parse_double(need(), "--hbm-twr-ns");
    } else if (key == "hbm-trtp-ns") {
        options.hbm_trtp_ns = parse_double(need(), "--hbm-trtp-ns");
    } else if (key == "hbm-tccd-s-cycles") {
        options.hbm_tccd_s_cycles =
            parse_u32(need(), "--hbm-tccd-s-cycles");
    } else if (key == "hbm-tccd-l-cycles") {
        options.hbm_tccd_l_cycles =
            parse_u32(need(), "--hbm-tccd-l-cycles");
    } else if (key == "hbm-trrd-s-ns") {
        options.hbm_trrd_s_ns = parse_double(need(), "--hbm-trrd-s-ns");
    } else if (key == "hbm-trrd-l-ns") {
        options.hbm_trrd_l_ns = parse_double(need(), "--hbm-trrd-l-ns");
    } else if (key == "hbm-tfaw-ns") {
        options.hbm_tfaw_ns = parse_double(need(), "--hbm-tfaw-ns");
    } else if (key == "hbm-twtr-s-ns") {
        options.hbm_twtr_s_ns = parse_double(need(), "--hbm-twtr-s-ns");
    } else if (key == "hbm-twtr-l-ns") {
        options.hbm_twtr_l_ns = parse_double(need(), "--hbm-twtr-l-ns");
    } else if (key == "hbm-trtw-ns") {
        options.hbm_trtw_ns = parse_double(need(), "--hbm-trtw-ns");
    } else if (key == "hbm-refresh") {
        options.hbm_refresh_enabled = parse_bool(need());
    } else if (key == "hbm-same-bank-refresh") {
        options.hbm_same_bank_refresh = parse_bool(need());
    } else if (key == "hbm-trefi-ns") {
        options.hbm_trefi_ns = parse_double(need(), "--hbm-trefi-ns");
    } else if (key == "hbm-trfc-ns") {
        options.hbm_trfc_ns = parse_double(need(), "--hbm-trfc-ns");
    } else if (key == "hbm-trfcsb-ns") {
        options.hbm_trfcsb_ns = parse_double(need(), "--hbm-trfcsb-ns");
    } else if (key == "hbm-queue-depth") {
        options.hbm_queue_depth = parse_u32(need(), "--hbm-queue-depth");
    } else if (key == "hbm-frfcfs-cap-ns") {
        options.hbm_frfcfs_cap_ns = parse_double(need(), "--hbm-frfcfs-cap-ns");
    } else if (key == "hbf-stacks") {
        options.hbf_stacks = parse_u32(need(), "--hbf-stacks");
    } else if (key == "hbf-channels") {
        options.hbf_channels = parse_u32(need(), "--hbf-channels");
    } else if (key == "hbf-dies-per-channel") {
        options.hbf_dies_per_channel = parse_u32(need(), "--hbf-dies-per-channel");
    } else if (key == "hbf-planes-per-die") {
        options.hbf_planes_per_die = parse_u32(need(), "--hbf-planes-per-die");
    } else if (key == "hbf-blocks-per-plane") {
        options.hbf_blocks_per_plane = parse_u32(need(), "--hbf-blocks-per-plane");
    } else if (key == "hbf-pages-per-block") {
        options.hbf_pages_per_block = parse_u32(need(), "--hbf-pages-per-block");
    } else if (key == "hbf-page-size") {
        options.hbf_page_size = parse_u64(need());
    } else if (key == "hbf-oob-bytes") {
        options.hbf_oob_bytes = parse_u64(need());
    } else if (key == "hbf-media-lanes-per-plane") {
        options.hbf_media_lanes_per_plane = parse_u32(need(), "--hbf-media-lanes-per-plane");
    } else if (key == "hbf-subarrays-per-plane") {
        options.hbf_subarrays_per_plane = parse_u32(need(), "--hbf-subarrays-per-plane");
    } else if (key == "hbf-page-buffer-banks-per-plane") {
        options.hbf_page_buffer_banks_per_plane =
            parse_u32(need(), "--hbf-page-buffer-banks-per-plane");
    } else if (key == "hbf-read-ns") {
        options.hbf_read_ns = parse_double(need(), "--hbf-read-ns");
    } else if (key == "hbf-program-ns") {
        options.hbf_program_ns = parse_double(need(), "--hbf-program-ns");
    } else if (key == "hbf-program-verify-ns") {
        options.hbf_program_verify_ns =
            parse_double(need(), "--hbf-program-verify-ns");
    } else if (key == "hbf-erase-ns") {
        options.hbf_erase_ns = parse_double(need(), "--hbf-erase-ns");
    } else if (key == "hbf-ecc-decode-latency-ns") {
        options.hbf_ecc_decode_latency_ns =
            parse_double(need(), "--hbf-ecc-decode-latency-ns");
    } else if (key == "hbf-ecc-encode-latency-ns") {
        options.hbf_ecc_encode_latency_ns =
            parse_double(need(), "--hbf-ecc-encode-latency-ns");
    } else if (key == "hbf-ecc-decode-raw-bw") {
        options.hbf_ecc_decode_raw_bw_GBps_per_die =
            parse_double(need(), "--hbf-ecc-decode-raw-bw");
    } else if (key == "hbf-ecc-encode-raw-bw") {
        options.hbf_ecc_encode_raw_bw_GBps_per_die =
            parse_double(need(), "--hbf-ecc-encode-raw-bw");
    } else if (key == "hbf-channel-bw") {
        options.hbf_channel_bw_GBps = parse_double(need(), "--hbf-channel-bw");
    } else if (key == "hbf-hbio-bw") {
        options.hbf_hbio_bw_GBps = parse_double(need(), "--hbf-hbio-bw");
    } else if (key == "hbf-tsv-bw") {
        options.hbf_tsv_bw_GBps = parse_double(need(), "--hbf-tsv-bw");
    } else if (key == "hbf-media-lane-bw") {
        options.hbf_media_lane_bw_GBps = parse_double(need(), "--hbf-media-lane-bw");
    } else if (key == "hbf-logic-sram-bw") {
        options.hbf_logic_sram_bw_GBps = parse_double(need(), "--hbf-logic-sram-bw");
    } else if (key == "hbf-page-buffer-bw") {
        options.hbf_page_buffer_bw_GBps = parse_double(need(), "--hbf-page-buffer-bw");
    } else if (key == "hbf-ctrl-dram-bytes") {
        options.hbf_ctrl_dram_bytes = parse_u64(need());
    } else if (key == "static-direct-hbm-bytes") {
        options.static_direct_hbm_bytes = parse_u64(need());
    } else if (key == "scenarios") {
        options.scenarios.clear();
        std::string list = need();
        std::size_t start = 0;
        while (start <= list.size()) {
            const auto comma = list.find(',', start);
            const auto token = list.substr(
                start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!token.empty()) {
                options.scenarios.push_back(token);
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
    } else if (key == "hbf-logic-scheduler-issue-ns") {
        options.hbf_logic_scheduler_issue_ns = parse_double(need(), "--hbf-logic-scheduler-issue-ns");
    } else if (key == "hbf-flash-tsu-issue-ns") {
        options.hbf_flash_tsu_issue_ns =
            parse_double(need(), "--hbf-flash-tsu-issue-ns");
    } else if (key == "hbf-batch-activation") {
        options.hbf_batch_activation = parse_bool(need());
    } else if (key == "hbf-read-buffer-pages") {
        options.hbf_read_buffer_pages = parse_u64(need());
    } else if (key == "hbf-page-read-queue-depth-per-stack") {
        options.hbf_page_read_queue_depth_per_stack = parse_u64(need());
    } else if (key == "hbf-ctrl-dram-latency-ns") {
        options.hbf_ctrl_dram_latency_ns = parse_double(need(), "--hbf-ctrl-dram-latency-ns");
    } else if (key == "hbf-ctrl-dram-issue-ns") {
        options.hbf_ctrl_dram_issue_ns =
            parse_double(need(), "--hbf-ctrl-dram-issue-ns");
    } else if (key == "hbf-gc-low-watermark-pages") {
        options.hbf_gc_low_watermark_pages = parse_u64(need());
    } else if (key == "hbf-gc-hard-watermark-pages") {
        options.hbf_gc_hard_watermark_pages = parse_u64(need());
    } else if (key == "hbf-gc-reserved-free-blocks-per-plane") {
        options.hbf_gc_reserved_free_blocks_per_plane = parse_u64(need());
    } else if (key == "hbf-gc-wear-leveling-weight") {
        options.hbf_gc_wear_leveling_weight = parse_double(need(), "--hbf-gc-wear-leveling-weight");
    } else if (key == "hbf-write-coalescing") {
        options.hbf_write_coalescing = parse_bool(need());
    } else if (key == "hbf-write-buffer-completion-requires-flush") {
        options.hbf_write_buffer_completion_requires_flush = parse_bool(need());
    } else if (key == "hbf-write-buffer-pages") {
        options.hbf_write_buffer_pages = parse_u64(need());
    } else if (key == "hbf-write-buffer-flush-threshold-pages") {
        options.hbf_write_buffer_flush_threshold_pages = parse_u64(need());
    } else if (key == "hbf-hbm-write-buffer-bytes") {
        options.hbf_hbm_write_buffer_bytes = parse_u64(need());
    } else if (key == "hbf-hbm-write-buffer-destage") {
        options.hbf_hbm_write_buffer_destage = need();
        if (options.hbf_hbm_write_buffer_destage != "deferred" &&
            options.hbf_hbm_write_buffer_destage != "streamed") {
            throw std::runtime_error(
                "--hbf-hbm-write-buffer-destage must be deferred or streamed");
        }
    } else if (key == "trace-mode") {
        options.trace_mode = parse_trace_mode_value(need());
    } else {
        throw std::runtime_error("unknown option: " + raw_key);
    }
}

void load_config_file(Options& options, const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open config file: " + path);
    }
    std::string line;
    std::size_t line_no = 0;
    std::set<std::string> seen_keys;
    while (std::getline(in, line)) {
        ++line_no;
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error(
                "config line " + std::to_string(line_no) + " must be key=value");
        }
        const auto key = trim(line.substr(0, eq));
        const auto value = trim(line.substr(eq + 1));
        if (key.empty()) {
            throw std::runtime_error("config line " + std::to_string(line_no) + " has empty key");
        }
        if (key == "config") {
            throw std::runtime_error(
                "config files cannot include other config files (line " +
                std::to_string(line_no) + ")");
        }
        if (!seen_keys.insert(key).second) {
            throw std::runtime_error(
                "duplicate config key " + key + " at line " +
                std::to_string(line_no));
        }
        apply_option(options, key, value);
    }
}

std::vector<TraceOp> load_ramulator_trace(const Options& options) {
    if (!options.trace_path) {
        throw std::runtime_error(
            "--trace is required unless a trace generator is used without simulation");
    }
    std::ifstream in(*options.trace_path);
    if (!in) {
        throw std::runtime_error("cannot open trace: " + *options.trace_path);
    }

    std::vector<TraceOp> ops;
    std::string line;
    std::size_t line_no = 0;
    std::set<std::string> unknown_labels;
    while (std::getline(in, line)) {
        ++line_no;
        const double arrival = static_cast<double>(ops.size()) * options.interarrival_ns;
        auto op = parse_trace_line(line, line_no, options.line_size, arrival, unknown_labels);
        if (!op) {
            continue;
        }
        ops.push_back(*op);
        if (options.max_ops != 0 && ops.size() >= options.max_ops) {
            break;
        }
    }
    if (ops.empty()) {
        throw std::runtime_error("trace contains no memory operations");
    }
    if (!unknown_labels.empty()) {
        std::cerr << "warning: trace labels matching no semantic kind (kept as labels, kind=unknown):";
        for (const auto& token : unknown_labels) {
            std::cerr << ' ' << token;
        }
        std::cerr << '\n';
    }
    return ops;
}

void ensure_parent_dir(const std::string& output_path) {
    const std::filesystem::path path(output_path);
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void finish_output(
    std::ofstream& out,
    const std::string& output_path,
    const char* description) {
    out.flush();
    if (!out) {
        throw std::runtime_error(
            std::string("failed while writing ") + description + ": " + output_path);
    }
}

void emit_ramulator_op(std::ostream& out, std::uint64_t addr, Op op) {
    out << "0x" << std::hex << addr << std::dec << ' '
        << (op == Op::Read ? 'R' : 'W') << '\n';
}

void emit_semantic_op(
    std::ostream& out,
    std::uint64_t addr,
    Op op,
    std::uint64_t bytes,
    SemanticKind kind,
    std::uint64_t layer) {
    out << "0x" << std::hex << addr << std::dec << ' '
        << (op == Op::Read ? 'R' : 'W') << ' '
        << bytes << ' '
        << hbfsim::physical::hybrid::to_string(kind)
        << " layer=" << layer
        << '\n';
}

void generate_llm_trace(const Options& options) {
    const bool semantic = options.generate_semantic_llm_path.has_value();
    const auto& output_path = semantic ? *options.generate_semantic_llm_path : *options.generate_llm_path;
    if (output_path.empty()) {
        throw std::runtime_error("internal error: missing generate path");
    }

    constexpr std::uint64_t layer_stride = 0x0100'0000ull;
    constexpr std::uint64_t token_stride = 0x0000'4000ull;
    constexpr std::uint64_t scratch_layer_stride = 0x0002'0000ull;
    constexpr std::uint64_t weight_lines = 128;
    constexpr std::uint64_t kv_lines = 32;
    constexpr std::uint64_t scratch_lines = 16;
    const auto last_layer = static_cast<std::uint64_t>(options.llm_layers - 1);
    const auto last_token = static_cast<std::uint64_t>(options.llm_tokens - 1);
    const auto checked_region_end = [&] (
        std::uint64_t base,
        std::uint64_t layer_offset,
        std::uint64_t token_offset,
        std::uint64_t lines,
        const char* name) {
        auto end = checked_add_u64(base, layer_offset, name);
        end = checked_add_u64(end, token_offset, name);
        const auto line_bytes = checked_mul_u64(lines, options.line_size, name);
        return checked_add_u64(end, line_bytes - 1, name);
    };
    (void)checked_region_end(
        options.llm_weight_base,
        checked_mul_u64(last_layer, layer_stride, "generated weight address range"),
        0,
        weight_lines,
        "generated weight address range");
    (void)checked_region_end(
        options.llm_kv_base,
        checked_mul_u64(last_layer, layer_stride, "generated KV address range"),
        checked_mul_u64(last_token, token_stride, "generated KV address range"),
        kv_lines,
        "generated KV address range");
    (void)checked_region_end(
        options.llm_scratch_base,
        checked_add_u64(
            checked_mul_u64(
                std::min<std::uint64_t>(last_token, 3),
                layer_stride,
                "generated scratch address range"),
            checked_mul_u64(
                last_layer,
                scratch_layer_stride,
                "generated scratch address range"),
            "generated scratch address range"),
        0,
        scratch_lines,
        "generated scratch address range");

    ensure_parent_dir(output_path);
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot write generated trace: " + output_path);
    }

    out << (semantic ?
            "# HBFSim generated semantic LLM-like trace\n" :
            "# HBFSim generated LLM-like Ramulator trace\n");
    out << "# llm_tokens=" << options.llm_tokens << '\n';
    out << "# llm_layers=" << options.llm_layers << '\n';
    out << "# llm_weight_base=0x" << std::hex << options.llm_weight_base << std::dec << '\n';
    out << "# llm_kv_base=0x" << std::hex << options.llm_kv_base << std::dec << '\n';
    out << "# llm_scratch_base=0x" << std::hex << options.llm_scratch_base << std::dec << '\n';

    for (std::size_t token = 0; token < options.llm_tokens; ++token) {
        for (std::size_t layer = 0; layer < options.llm_layers; ++layer) {
            const auto execution_layer = checked_add_u64(
                checked_mul_u64(token, options.llm_layers, "generated layer id"),
                layer,
                "generated layer id");
            const auto layer_weight = options.llm_weight_base + layer * layer_stride;
            for (std::size_t line = 0; line < weight_lines; ++line) {
                if (semantic) {
                    emit_semantic_op(
                        out,
                        layer_weight + line * options.line_size,
                        Op::Read,
                        options.line_size,
                        SemanticKind::ModelWeights,
                        execution_layer);
                } else {
                    emit_ramulator_op(out, layer_weight + line * options.line_size, Op::Read);
                }
            }

            const auto layer_kv = options.llm_kv_base + layer * layer_stride;
            const auto history = std::min<std::size_t>(token, 32);
            for (std::size_t prev = token - history; prev < token; ++prev) {
                for (std::size_t line = 0; line < kv_lines; ++line) {
                    const auto addr = layer_kv + prev * token_stride + line * options.line_size;
                    if (semantic) {
                        emit_semantic_op(
                            out,
                            addr,
                            Op::Read,
                            options.line_size,
                            SemanticKind::SharedContext,
                            execution_layer);
                    } else {
                        emit_ramulator_op(out, addr, Op::Read);
                    }
                }
            }
            for (std::size_t line = 0; line < kv_lines; ++line) {
                const auto addr = layer_kv + token * token_stride + line * options.line_size;
                if (semantic) {
                    emit_semantic_op(
                        out,
                        addr,
                        Op::Write,
                        options.line_size,
                        SemanticKind::GeneratedContext,
                        execution_layer);
                } else {
                    emit_ramulator_op(out, addr, Op::Write);
                }
            }

            const auto scratch = options.llm_scratch_base +
                (token % 4) * layer_stride + layer * scratch_layer_stride;
            for (std::size_t line = 0; line < scratch_lines; ++line) {
                const auto addr = scratch + line * options.line_size;
                if (semantic) {
                    emit_semantic_op(
                        out, addr, Op::Write, options.line_size,
                        SemanticKind::Scratch, execution_layer);
                    emit_semantic_op(
                        out, addr, Op::Read, options.line_size,
                        SemanticKind::Scratch, execution_layer);
                } else {
                    emit_ramulator_op(out, addr, Op::Write);
                    emit_ramulator_op(out, addr, Op::Read);
                }
            }
        }
    }
    finish_output(out, output_path, "generated trace");
}

TraceMode effective_trace_mode(const Options& options) {
    if (options.trace_mode != TraceMode::Off) {
        return options.trace_mode;
    }
    return options.chrome_trace_path ? TraceMode::Full : TraceMode::Off;
}

std::uint64_t hbf_capacity_unit_bytes(const Options& options) {
    auto value = static_cast<std::uint64_t>(options.hbf_stacks);
    value = checked_mul_u64(value, options.hbf_channels, "HBF capacity unit");
    value = checked_mul_u64(value, options.hbf_dies_per_channel, "HBF capacity unit");
    value = checked_mul_u64(value, options.hbf_planes_per_die, "HBF capacity unit");
    value = checked_mul_u64(value, options.hbf_pages_per_block, "HBF capacity unit");
    value = checked_mul_u64(value, options.hbf_page_size, "HBF capacity unit");
    return value;
}

std::uint64_t hbf_capacity_bytes(const Options& options) {
    return checked_mul_u64(
        hbf_capacity_unit_bytes(options),
        options.hbf_blocks_per_plane,
        "HBF capacity");
}

std::uint64_t hbm_cooperative_region_base_addr(const Options& options) {
    if (options.hbf_hbm_write_buffer_bytes > options.hbm_capacity_bytes) {
        throw std::runtime_error("cooperative write region exceeds HBM capacity");
    }
    return options.hbm_capacity_bytes - options.hbf_hbm_write_buffer_bytes;
}

std::uint64_t trace_unique_line_bytes(
    const std::vector<TraceOp>& ops,
    std::uint64_t line_size) {
    if (line_size == 0) {
        throw std::runtime_error("trace line size must be positive");
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> intervals;
    intervals.reserve(ops.size());
    for (const auto& op : ops) {
        const auto first = op.addr / line_size;
        const auto last = checked_add_u64(
            op.addr,
            op.bytes - 1,
            "trace unique-line range") / line_size;
        intervals.emplace_back(first, last);
    }
    if (intervals.empty()) {
        return 0;
    }
    std::sort(intervals.begin(), intervals.end());
    std::uint64_t unique_lines = 0;
    auto current_first = intervals.front().first;
    auto current_last = intervals.front().second;
    const auto add_interval = [&unique_lines](
                                  std::uint64_t first,
                                  std::uint64_t last) {
        const auto span_minus_one = last - first;
        const auto span = checked_add_u64(
            span_minus_one, 1, "trace unique-line interval length");
        unique_lines = checked_add_u64(
            unique_lines, span, "trace unique-line count");
    };
    for (std::size_t index = 1; index < intervals.size(); ++index) {
        const auto [next_first, next_last] = intervals[index];
        const bool adjacent = current_last !=
                std::numeric_limits<std::uint64_t>::max() &&
            next_first == current_last + 1;
        if (next_first <= current_last || adjacent) {
            current_last = std::max(current_last, next_last);
            continue;
        }
        add_interval(current_first, current_last);
        current_first = next_first;
        current_last = next_last;
    }
    add_interval(current_first, current_last);
    return checked_mul_u64(
        unique_lines, line_size, "trace unique-line footprint");
}

void apply_capacity_targets(Options& options) {
    if (options.hbm_capacity_explicit && !options.flat_hbm_bytes_explicit) {
        options.flat_hbm_bytes = options.hbm_capacity_bytes;
    }
    if (options.hbf_capacity_bytes_target && options.hbf_capacity_ratio) {
        throw std::runtime_error(
            "--hbf-capacity-bytes and --hbf-capacity-ratio are mutually exclusive");
    }
    if (options.hbf_capacity_ratio) {
        if (!(*options.hbf_capacity_ratio > 0.0) || !std::isfinite(*options.hbf_capacity_ratio)) {
            throw std::runtime_error("--hbf-capacity-ratio must be positive and finite");
        }
        const auto target = static_cast<long double>(options.hbm_capacity_bytes) *
            static_cast<long double>(*options.hbf_capacity_ratio);
        if (target > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
            throw std::runtime_error("--hbf-capacity-ratio target exceeds uint64_t range");
        }
        options.hbf_capacity_bytes_target = static_cast<std::uint64_t>(std::ceil(target));
    }
    if (options.hbf_capacity_bytes_target) {
        const auto unit = hbf_capacity_unit_bytes(options);
        const auto blocks = std::max<std::uint64_t>(
            1,
            ceil_div_u64(*options.hbf_capacity_bytes_target, unit));
        if (blocks > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("derived --hbf-blocks-per-plane exceeds uint32_t range");
        }
        options.hbf_blocks_per_plane = static_cast<std::uint32_t>(blocks);
    }
}

HbmConfig make_hbm_config(const Options& options) {
    HbmConfig cfg;
    cfg.capacity_bytes = options.hbm_capacity_bytes;
    cfg.stacks = options.hbm_stacks;
    cfg.channels_per_stack = options.hbm_channels;
    cfg.pseudo_channels_per_channel = options.hbm_pseudo_channels;
    cfg.bank_groups_per_pseudo_channel =
        options.hbm_bank_groups_per_pseudo_channel;
    cfg.banks_per_group = options.hbm_banks_per_group;
    cfg.channel_row_size_bytes = options.hbm_channel_row_size_bytes;
    cfg.channel_width_bits = options.hbm_channel_width_bits;
    cfg.burst_length = options.hbm_burst_length;
    cfg.pin_rate_Gbps = options.hbm_pin_rate_Gbps;
    cfg.data_rate_per_command_clock = options.hbm_data_rate_per_command_clock;
    cfg.address_mapping_ns = options.hbm_address_mapping_ns;
    cfg.tRCDRD_ns = options.hbm_trcdrd_ns;
    cfg.tRCDWR_ns = options.hbm_trcdwr_ns;
    cfg.tCL_ns = options.hbm_tcl_ns;
    cfg.tCWL_ns = options.hbm_tcwl_ns;
    cfg.tRP_ns = options.hbm_trp_ns;
    cfg.tRAS_ns = options.hbm_tras_ns;
    cfg.tRC_ns = options.hbm_trc_ns;
    cfg.tWR_ns = options.hbm_twr_ns;
    cfg.tRTP_ns = options.hbm_trtp_ns;
    cfg.tCCD_S_cycles = options.hbm_tccd_s_cycles;
    cfg.tCCD_L_cycles = options.hbm_tccd_l_cycles;
    cfg.tRRD_S_ns = options.hbm_trrd_s_ns;
    cfg.tRRD_L_ns = options.hbm_trrd_l_ns;
    cfg.tFAW_ns = options.hbm_tfaw_ns;
    cfg.tWTR_S_ns = options.hbm_twtr_s_ns;
    cfg.tWTR_L_ns = options.hbm_twtr_l_ns;
    cfg.tRTW_ns = options.hbm_trtw_ns;
    cfg.refresh_enabled = options.hbm_refresh_enabled;
    cfg.same_bank_refresh = options.hbm_same_bank_refresh;
    cfg.tREFI_ns = options.hbm_trefi_ns;
    cfg.tRFC_ns = options.hbm_trfc_ns;
    cfg.tRFCsb_ns = options.hbm_trfcsb_ns;
    cfg.queue_depth = options.hbm_queue_depth;
    cfg.frfcfs_cap_ns = options.hbm_frfcfs_cap_ns;
    return cfg;
}

HbfConfig make_hbf_config(const Options& options) {
    HbfConfig cfg;
    cfg.stacks = options.hbf_stacks;
    cfg.channels_per_stack = options.hbf_channels;
    cfg.dies_per_channel = options.hbf_dies_per_channel;
    cfg.planes_per_die = options.hbf_planes_per_die;
    cfg.blocks_per_plane = options.hbf_blocks_per_plane;
    cfg.pages_per_block = options.hbf_pages_per_block;
    cfg.page_size_bytes = options.hbf_page_size;
    cfg.oob_bytes_per_page = options.hbf_oob_bytes;
    cfg.media_lanes_per_plane = options.hbf_media_lanes_per_plane;
    cfg.subarrays_per_plane = options.hbf_subarrays_per_plane;
    cfg.page_buffer_banks_per_plane = options.hbf_page_buffer_banks_per_plane;
    cfg.t_read_page_ns = options.hbf_read_ns;
    cfg.t_program_page_ns = options.hbf_program_ns;
    cfg.t_program_verify_ns = options.hbf_program_verify_ns;
    cfg.t_erase_block_ns = options.hbf_erase_ns;
    cfg.ecc_decode_latency_ns = options.hbf_ecc_decode_latency_ns;
    cfg.ecc_encode_latency_ns = options.hbf_ecc_encode_latency_ns;
    cfg.ecc_decode_raw_bandwidth_GBps_per_die =
        options.hbf_ecc_decode_raw_bw_GBps_per_die;
    cfg.ecc_encode_raw_bandwidth_GBps_per_die =
        options.hbf_ecc_encode_raw_bw_GBps_per_die;
    cfg.channel_bandwidth_GBps = options.hbf_channel_bw_GBps;
    cfg.hb_io_bandwidth_GBps = options.hbf_hbio_bw_GBps;
    cfg.tsv_bandwidth_GBps = options.hbf_tsv_bw_GBps;
    cfg.media_lane_bandwidth_GBps = options.hbf_media_lane_bw_GBps;
    cfg.logic_sram_bandwidth_GBps = options.hbf_logic_sram_bw_GBps;
    cfg.page_buffer_bandwidth_GBps = options.hbf_page_buffer_bw_GBps;
    cfg.ctrl_dram_bytes = options.hbf_ctrl_dram_bytes;
    cfg.ctrl_dram_latency_ns = options.hbf_ctrl_dram_latency_ns;
    cfg.ctrl_dram_issue_ns = options.hbf_ctrl_dram_issue_ns;
    cfg.flash_tsu_issue_ns = options.hbf_flash_tsu_issue_ns;
    cfg.logic_scheduler_issue_ns = options.hbf_logic_scheduler_issue_ns;
    cfg.batch_activation = options.hbf_batch_activation;
    cfg.read_buffer_pages = options.hbf_read_buffer_pages;
    cfg.page_read_queue_depth_per_stack =
        options.hbf_page_read_queue_depth_per_stack;
    cfg.gc_low_watermark_pages = options.hbf_gc_low_watermark_pages;
    cfg.gc_hard_watermark_pages = options.hbf_gc_hard_watermark_pages;
    cfg.gc_reserved_free_blocks_per_plane = options.hbf_gc_reserved_free_blocks_per_plane;
    cfg.gc_wear_leveling_weight = options.hbf_gc_wear_leveling_weight;
    cfg.write_coalescing_enabled = options.hbf_write_coalescing;
    cfg.write_buffer_completion_requires_flush =
        options.hbf_write_buffer_completion_requires_flush;
    cfg.write_buffer_pages = options.hbf_write_buffer_pages;
    cfg.write_buffer_flush_threshold_pages = options.hbf_write_buffer_flush_threshold_pages;
    if (cfg.ctrl_dram_bytes == 0) {
        cfg.ctrl_dram_bytes =
            hbfsim::physical::hbf::derive_resident_mapping_capacity(cfg)
                .total_bytes;
    }
    return cfg;
}

ExternalBackingConfig make_external_backing_config(const Options& options) {
    ExternalBackingConfig cfg;
    switch (options.external_backing_kind) {
    case ExternalBackingKind::OnPackageLpddr:
        cfg = hbfsim::physical::external::on_package_lpddr_profile();
        break;
    case ExternalBackingKind::CxlMemory:
        cfg = hbfsim::physical::external::cxl_memory_profile();
        break;
    case ExternalBackingKind::NvmeSsd:
        cfg = hbfsim::physical::external::nvme_ssd_profile();
        break;
    }
    if (options.external_backing_capacity_bytes) {
        cfg.capacity_bytes = *options.external_backing_capacity_bytes;
    }
    if (options.external_backing_page_size) {
        cfg.page_size_bytes = *options.external_backing_page_size;
    }
    if (options.external_backing_media_channels) {
        cfg.media_channels = *options.external_backing_media_channels;
    }
    if (options.external_backing_max_outstanding_requests) {
        cfg.max_outstanding_requests =
            *options.external_backing_max_outstanding_requests;
    }
    if (options.external_backing_controller_issue_ns) {
        cfg.controller_issue_ns =
            *options.external_backing_controller_issue_ns;
    }
    if (options.external_backing_controller_processing_ns) {
        cfg.controller_processing_ns =
            *options.external_backing_controller_processing_ns;
    }
    if (options.external_backing_media_read_latency_ns) {
        cfg.media_read_latency_ns =
            *options.external_backing_media_read_latency_ns;
    }
    if (options.external_backing_media_write_latency_ns) {
        cfg.media_write_latency_ns =
            *options.external_backing_media_write_latency_ns;
    }
    if (options.external_backing_media_read_bw_GBps) {
        cfg.media_read_bandwidth_GBps =
            *options.external_backing_media_read_bw_GBps;
    }
    if (options.external_backing_media_write_bw_GBps) {
        cfg.media_write_bandwidth_GBps =
            *options.external_backing_media_write_bw_GBps;
    }
    if (options.external_backing_m2s_bw_GBps) {
        cfg.m2s_bandwidth_GBps =
            *options.external_backing_m2s_bw_GBps;
    }
    if (options.external_backing_s2m_bw_GBps) {
        cfg.s2m_bandwidth_GBps =
            *options.external_backing_s2m_bw_GBps;
    }
    if (options.external_backing_one_way_propagation_ns) {
        cfg.one_way_propagation_ns =
            *options.external_backing_one_way_propagation_ns;
    }
    if (options.external_backing_command_bytes) {
        cfg.command_bytes = *options.external_backing_command_bytes;
    }
    if (options.external_backing_completion_bytes) {
        cfg.completion_bytes =
            *options.external_backing_completion_bytes;
    }
    return cfg;
}

// Adapt a library-side composition run into the tool's per-scenario record.
ScenarioResult from_direct_run(CompositionRunResult&& run, std::string name) {
    ScenarioResult result;
    result.name = std::move(name);
    result.has_hbm = run.has_hbm;
    result.has_hbf = run.has_hbf;
    result.service_latencies_ns = std::move(run.service_latencies_ns);
    result.offered_latencies_ns = std::move(run.offered_latencies_ns);
    result.source_latencies_ns = std::move(run.source_latencies_ns);
    result.completions = std::move(run.completions);
    result.warnings = std::move(run.warnings);
    result.ops = run.ops;
    result.reads = run.reads;
    result.writes = run.writes;
    result.logical_bytes = run.logical_bytes;
    result.hbm_user_accesses = run.hbm_user_accesses;
    result.hbf_user_accesses = run.hbf_user_accesses;
    result.hbf_direct_user_ops = run.hbf_direct_user_ops;
    result.hbm_background_accesses = run.hbm_background_accesses;
    result.hbf_background_accesses = run.hbf_background_accesses;
    result.hbm_accesses = checked_add_u64(
        run.hbm_user_accesses,
        run.hbm_background_accesses,
        "scenario HBM access count");
    result.hbf_accesses = checked_add_u64(
        run.hbf_user_accesses,
        run.hbf_background_accesses,
        "scenario HBF access count");
    result.hbf_static_read_bytes = run.hbf_static_read_bytes;
    result.hbm_write_buffer_user_write_bytes =
        run.hbm_write_buffer_user_write_bytes;
    result.hbm_write_buffer_destaged_bytes = run.hbm_write_buffer_destaged_bytes;
    result.hbm_write_buffer_peak_bytes = run.hbm_write_buffer_peak_bytes;
    result.hbm_write_buffer_full_waits = run.hbm_write_buffer_full_waits;
    result.hbm_write_buffer_wait_ns = run.hbm_write_buffer_wait_ns;
    result.cooperative_write_controller_present =
        run.cooperative_write_controller_present;
    result.base_die_link_stats = run.base_die_link_stats;
    result.phase_barriers = run.phase_barriers;
    result.phase_dependency_waited_ops =
        run.phase_dependency_waited_ops;
    result.phase_dependency_wait_work_ns =
        run.phase_dependency_wait_work_ns;
    result.phase_dependency_max_wait_ns =
        run.phase_dependency_max_wait_ns;
    result.front_end_admission_waited_ops = run.front_end_admission_waited_ops;
    result.front_end_admission_wait_work_ns = run.front_end_admission_wait_work_ns;
    result.front_end_admission_max_wait_ns = run.front_end_admission_max_wait_ns;
    result.finish_ns = run.finish_ns;
    result.user_finish_ns = run.user_finish_ns;
    result.first_arrival_ns = run.first_offered_arrival_ns;
    result.last_arrival_ns = run.last_offered_arrival_ns;
    result.owns_arrival_frontier = true;
    result.hbm_stats = run.hbm_stats;
    result.hbf_stats = run.hbf_stats;
    result.hbf_logical_read_bytes = run.has_hbf ?
        run.hbf_stats.logical_read_bytes :
        0;
    result.address_heatmap = std::move(run.address_heatmap);
    return result;
}

DirectRunKnobs make_direct_run_knobs(
    const Options& options,
    const std::vector<MemoryRequest>* initial_image_requests = nullptr) {
    return DirectRunKnobs{
        .max_outstanding_requests = options.max_outstanding_requests,
        .max_hbm_outstanding_requests =
            options.max_hbm_outstanding_requests,
        .max_hbf_outstanding_requests =
            options.max_hbf_outstanding_requests,
        .hbm_write_buffer_bytes = options.hbf_hbm_write_buffer_bytes,
        .destage_policy = options.hbf_hbm_write_buffer_destage == "streamed" ?
            DirectRunKnobs::DestagePolicy::Streamed :
            DirectRunKnobs::DestagePolicy::Deferred,
        .base_die_link = BaseDieLinkConfig{
            .read_bandwidth_GBps = options.base_die_link_read_bw_GBps,
            .write_bandwidth_GBps = options.base_die_link_write_bw_GBps,
            .latency_ns = options.base_die_link_latency_ns,
        },
        .trace = {
            .mode = effective_trace_mode(options),
            .retain_completion_diagnostics =
                options.chrome_trace_path.has_value(),
        },
        .retain_completions = options.chrome_trace_path.has_value(),
        .address_heatmap_bins = options.address_heatmap_bins,
        .initial_image_requests = initial_image_requests,
    };
}

template <typename RequestSource>
ScenarioResult run_direct(
    const DirectPolicy& policy,
    std::string scenario_name,
    const RequestSource& requests,
    const Options& options,
    const std::vector<MemoryRequest>* initial_image_requests = nullptr) {
    auto run = run_direct_composition(
        policy,
        make_hbm_config(options),
        make_hbf_config(options),
        requests,
        make_direct_run_knobs(options, initial_image_requests));
    return from_direct_run(std::move(run), std::move(scenario_name));
}

BehavioralTieringConfig make_behavioral_tiering_config(
    const Options& options,
    BehavioralAdmissionPolicy policy,
    const std::vector<MemoryRequest>* initial_image_requests) {
    return BehavioralTieringConfig{
        .hbm = make_hbm_config(options),
        .hbf = make_hbf_config(options),
        .base_die_link = BaseDieLinkConfig{
            .read_bandwidth_GBps = options.base_die_link_read_bw_GBps,
            .write_bandwidth_GBps = options.base_die_link_write_bw_GBps,
            .latency_ns = options.base_die_link_latency_ns,
        },
        .admission_policy = policy,
        .hbm_tier_bytes = options.behavioral_hbm_bytes == 0 ?
            options.hbm_capacity_bytes :
            options.behavioral_hbm_bytes,
        .promotion_threshold =
            options.behavioral_promotion_threshold,
        .history_capacity_pages =
            options.behavioral_history_pages,
        .max_outstanding_requests =
            options.max_outstanding_requests,
        .address_heatmap_bins = options.address_heatmap_bins,
        .trace = {
            .mode = effective_trace_mode(options),
            .retain_completion_diagnostics =
                options.chrome_trace_path.has_value(),
        },
        .retain_completions =
            options.chrome_trace_path.has_value(),
        .retain_decisions = false,
        .initial_image_requests = initial_image_requests,
    };
}

ScenarioResult run_behavioral_tiering(
    const std::vector<MemoryRequest>& requests,
    const Options& options,
    BehavioralAdmissionPolicy policy,
    std::string scenario_name,
    const std::vector<MemoryRequest>* initial_image_requests) {
    auto run = run_behavioral_tiering_composition(
        make_behavioral_tiering_config(
            options, policy, initial_image_requests),
        requests);
    auto result = from_direct_run(
        std::move(run.composition), std::move(scenario_name));
    result.background_hbf_writes =
        run.placement.dirty_writeback_pages;
    result.hbf_backing_write_bytes =
        run.placement.dirty_writeback_bytes;
    result.hbm_foreground_bytes =
        run.placement.hbm_foreground_bytes;
    result.behavioral_tiering_stats = std::move(run.placement);
    return result;
}

std::vector<MemoryRequest> make_memory_requests(const std::vector<TraceOp>& ops) {
    std::vector<MemoryRequest> requests;
    requests.reserve(ops.size());
    for (std::size_t i = 0; i < ops.size(); ++i) {
        requests.push_back(MemoryRequest{
            .id = "op" + std::to_string(i),
            .op = ops[i].op,
            .addr = ops[i].addr,
            .bytes = ops[i].bytes,
            .arrival_ns = ops[i].arrival_ns,
            .index = i,
            .kind = ops[i].kind,
            .label = ops[i].label,
            .phase = ops[i].phase,
            .layer = ops[i].layer,
            .compute_ns = ops[i].compute_ns,
        });
    }
    return requests;
}

LayerStreamingConfig make_layer_streaming_config(
    const Options& options,
    BackingTier backing) {
    LayerStreamingConfig cfg;
    cfg.hbm = make_hbm_config(options);
    cfg.backing = backing;
    cfg.hbf = make_hbf_config(options);
    cfg.external_backing = make_external_backing_config(options);
    cfg.base_die_link = BaseDieLinkConfig{
        .read_bandwidth_GBps = options.base_die_link_read_bw_GBps,
        .write_bandwidth_GBps = options.base_die_link_write_bw_GBps,
        .latency_ns = options.base_die_link_latency_ns,
    };
    cfg.layer_buffer_bytes = options.layer_buffer_bytes;
    if (options.explicit_residency_contract) {
        cfg.residency_contract = LayerResidencyContract{
            .page_size_bytes =
                *options.residency_page_size_bytes,
            .unique_resident_footprint_bytes =
                *options.residency_unique_footprint_bytes,
            .immutable_weight_bytes =
                *options.residency_immutable_weight_bytes,
            .immutable_weight_pages =
                *options.residency_immutable_weight_pages,
            .static_weight_resident_pages =
                *options.residency_static_weight_pages,
            .runtime_overhead_bytes =
                *options.residency_runtime_overhead_bytes,
            .block_table_bytes =
                *options.residency_block_table_bytes,
            .active_buffer_bytes_per_slot =
                *options.residency_active_buffer_bytes,
            .kv_region_begin =
                *options.residency_kv_region_begin,
            .kv_block_stride_bytes =
                *options.residency_kv_block_stride_bytes,
            .logical_kv_blocks =
                *options.residency_logical_kv_blocks,
            .hot_kv_blocks =
                *options.residency_hot_kv_blocks,
        };
    }
    cfg.max_outstanding_requests = options.max_outstanding_requests;
    cfg.address_heatmap_bins = options.address_heatmap_bins;
    cfg.trace = {
        .mode = effective_trace_mode(options),
        .retain_completion_diagnostics =
            options.chrome_trace_path.has_value(),
    };
    return cfg;
}

ScenarioResult run_layer_streaming(
    const std::vector<MemoryRequest>& requests,
    const Options& options,
    BackingTier backing) {
    ScenarioResult result;
    result.name = backing == BackingTier::Hbf ?
        kLayerStreamingScenario : kExternalLayerStreamingScenario;
    result.has_hbm = true;
    result.has_hbf = backing == BackingTier::Hbf;
    result.has_external_backing = backing == BackingTier::External;

    LayerStreamingComposition system(
        make_layer_streaming_config(options, backing));
    auto hybrid = system.run(requests);

    result.service_latencies_ns =
        std::move(hybrid.service_latencies_ns);
    result.offered_latencies_ns =
        std::move(hybrid.offered_latencies_ns);
    result.source_latencies_ns =
        std::move(hybrid.source_latencies_ns);
    result.completions = std::move(hybrid.completions);
    result.warnings = std::move(hybrid.warnings);
    result.ops = hybrid.ops;
    result.reads = hybrid.reads;
    result.writes = hybrid.writes;
    result.logical_bytes = hybrid.logical_bytes;
    result.hbm_user_accesses = hybrid.hbm_user_accesses;
    result.hbf_user_accesses = hybrid.hbf_user_accesses;
    result.hbm_background_accesses = hybrid.hbm_background_accesses;
    result.hbf_background_accesses = hybrid.hbf_background_accesses;
    result.external_background_accesses =
        hybrid.external_background_accesses;
    result.hbm_accesses = checked_add_u64(
        result.hbm_user_accesses,
        result.hbm_background_accesses,
        "layer-streaming HBM access count");
    result.hbf_accesses = checked_add_u64(
        result.hbf_user_accesses,
        result.hbf_background_accesses,
        "layer-streaming HBF access count");
    result.external_accesses = hybrid.external_background_accesses;
    result.layer_streaming_stats = hybrid.streaming_stats;
    result.background_hbf_writes = hybrid.background_hbf_writes;
    result.hbf_static_read_bytes = hybrid.hbf_static_read_bytes;
    result.hbf_logical_read_bytes = hybrid.hbf_logical_read_bytes;
    result.hbf_backing_write_bytes = hybrid.hbf_backing_write_bytes;
    result.external_backing_read_bytes =
        hybrid.external_backing_read_bytes;
    result.external_backing_write_bytes =
        hybrid.external_backing_write_bytes;
    result.hbm_foreground_bytes = hybrid.hbm_foreground_bytes;
    result.hbm_streaming_write_bytes = hybrid.hbm_streaming_write_bytes;
    result.base_die_link_stats = hybrid.base_die_link_stats;
    result.front_end_admission_waited_ops = hybrid.front_end_admission_waited_ops;
    result.front_end_admission_wait_work_ns = hybrid.front_end_admission_wait_work_ns;
    result.front_end_admission_max_wait_ns = hybrid.front_end_admission_max_wait_ns;
    result.first_arrival_ns = hybrid.first_offered_arrival_ns;
    result.last_arrival_ns = hybrid.last_offered_arrival_ns;
    result.phase_barriers = hybrid.phase_barriers;
    result.phase_dependency_waited_ops =
        hybrid.phase_dependency_waited_ops;
    result.phase_dependency_wait_work_ns =
        hybrid.phase_dependency_wait_work_ns;
    result.phase_dependency_max_wait_ns =
        hybrid.phase_dependency_max_wait_ns;
    result.owns_arrival_frontier = true;
    result.finish_ns = hybrid.finish_ns;
    result.user_finish_ns = hybrid.user_finish_ns;
    result.hbm_stats = hybrid.hbm_stats;
    result.hbf_stats = hybrid.hbf_stats;
    result.external_backing_stats = hybrid.external_backing_stats;
    result.address_heatmap = std::move(hybrid.address_heatmap);
    return result;
}

double percentile_sorted(const std::vector<double>& values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    const auto pos = (p / 100.0) * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    if (lo == hi) {
        return values[lo];
    }
    const auto frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

std::string fixed(double value, int precision = 2) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

struct WallClockSnapshot {
    double trace_origin_ns = 0.0;
    double last_offered_arrival_ns = 0.0;
    double last_user_completion_ns = 0.0;
    double quiescent_finish_ns = 0.0;
    double offered_arrival_span_ns = 0.0;
    double post_offer_user_completion_tail_ns = 0.0;
    double user_completion_span_ns = 0.0;
    double drain_tail_ns = 0.0;
    double makespan_ns = 0.0;
};

struct LatencyWorkSnapshot {
    std::uint64_t user_count = 0;
    double service_to_user_completion_sum_work_ns = 0.0;
    double offered_to_user_completion_sum_work_ns = 0.0;
    double source_to_user_completion_sum_work_ns = 0.0;
    // Primary online latency distribution: offered arrival through parent
    // completion, including front-end transaction-credit wait.
    double average_ns = 0.0;
    double p50_ns = 0.0;
    double p95_ns = 0.0;
    double max_ns = 0.0;
    double service_average_ns = 0.0;
    double service_p50_ns = 0.0;
    double service_p95_ns = 0.0;
    double service_max_ns = 0.0;
    double source_average_ns = 0.0;
    double source_p50_ns = 0.0;
    double source_p95_ns = 0.0;
    double source_max_ns = 0.0;
    std::uint64_t phase_barriers = 0;
    std::uint64_t phase_dependency_waited_ops = 0;
    double phase_dependency_wait_work_ns = 0.0;
    double phase_dependency_max_wait_ns = 0.0;
    std::uint64_t front_end_admission_waited_ops = 0;
    double front_end_admission_wait_work_ns = 0.0;
    double front_end_admission_max_wait_ns = 0.0;
};

struct BaseDieLinkStageWorkSnapshot {
    double read_queue_wait_work_ns = 0.0;
    double write_queue_wait_work_ns = 0.0;
    double read_serialization_work_ns = 0.0;
    double write_serialization_work_ns = 0.0;
    double read_fixed_latency_work_ns = 0.0;
    double write_fixed_latency_work_ns = 0.0;
};

struct LayerStreamingControllerStageWorkSnapshot {
    bool present = false;
    double backing_admission_wait_work_ns = 0.0;
    double backing_admission_max_wait_ns = 0.0;
    double user_wait_work_ns = 0.0;
    double user_max_wait_ns = 0.0;
    double exposed_prefetch_ns = 0.0;
    double hidden_prefetch_ns = 0.0;
    double buffer_reuse_wait_work_ns = 0.0;
};

struct CooperativeWriteControllerStageWorkSnapshot {
    bool present = false;
    std::uint64_t full_waited_ops = 0;
    double full_wait_work_ns = 0.0;
};

struct ResourceBusyMetric {
    double busy_ns = 0.0;
    std::uint64_t resource_count = 0;
    double active_span_ns = 0.0;
    double capacity_time_ns = 0.0;
    double utilization = 0.0;
};

struct ResourceBusySnapshot {
    ResourceBusyMetric hbm_data_bus;
    ResourceBusyMetric external_controller;
    ResourceBusyMetric external_media;
    ResourceBusyMetric external_link_m2s;
    ResourceBusyMetric external_link_s2m;
    ResourceBusyMetric hbf_logic_ingress;
    ResourceBusyMetric hbf_mapping_dram_issue;
    ResourceBusyMetric hbf_plane_media;
    ResourceBusyMetric hbf_media_lane;
    ResourceBusyMetric hbf_subarray;
    ResourceBusyMetric hbf_page_buffer_bank;
    ResourceBusyMetric hbf_flash_source_queue;
    ResourceBusyMetric hbf_channel_command;
    ResourceBusyMetric hbf_channel_data;
    ResourceBusyMetric hbf_tsv;
    ResourceBusyMetric hbf_sram;
    ResourceBusyMetric hbf_hbio_command;
    ResourceBusyMetric hbf_hbio_data;
    ResourceBusyMetric hbf_sequencer;
    ResourceBusyMetric hbf_ecc_issue;
    ResourceBusyMetric base_die_link_read;
    ResourceBusyMetric base_die_link_write;
};

struct TimeBreakdownSnapshot {
    WallClockSnapshot wall_clock;
    LatencyWorkSnapshot latency_work;
    Breakdown hbm_stage_work;
    Breakdown hbf_stage_work;
    Breakdown external_backing_stage_work;
    BaseDieLinkStageWorkSnapshot base_die_link_stage_work;
    LayerStreamingControllerStageWorkSnapshot
        layer_streaming_controller_stage_work;
    CooperativeWriteControllerStageWorkSnapshot
        cooperative_write_controller_stage_work;
    ResourceBusySnapshot resource_busy;
};

double sum_latency_work_ns(const std::vector<double>& values) {
    long double total = 0.0L;
    for (const auto value : values) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error("cannot sum a non-finite or negative latency");
        }
        total += static_cast<long double>(value);
    }
    if (total > static_cast<long double>(std::numeric_limits<double>::max())) {
        throw std::runtime_error("latency work exceeds finite double range");
    }
    return static_cast<double>(total);
}

LatencyDistribution finalize_latency_samples(
    std::vector<double>& samples,
    std::uint64_t expected_count,
    bool release_samples,
    std::string_view name) {
    if (samples.size() != expected_count) {
        throw std::runtime_error(
            "cannot finalize " + std::string(name) +
            " latency distribution with a count mismatch");
    }
    LatencyDistribution distribution;
    distribution.count = samples.size();
    // Accumulate once in extended precision, then derive the mean from the
    // reported sum.  Keeping two independent floating-point reductions made
    // sum/count diverge from the emitted average on long traces.
    distribution.sum_work_ns = sum_latency_work_ns(samples);
    distribution.average_ns = distribution.count == 0 ? 0.0 :
        distribution.sum_work_ns / static_cast<double>(distribution.count);
    std::sort(samples.begin(), samples.end());
    distribution.p50_ns = percentile_sorted(samples, 50.0);
    distribution.p95_ns = percentile_sorted(samples, 95.0);
    distribution.max_ns = percentile_sorted(samples, 100.0);
    if (release_samples) {
        std::vector<double>().swap(samples);
    }
    return distribution;
}

void finalize_latency_distributions(
    ScenarioResult& result,
    bool release_samples) {
    result.service_latency_distribution = finalize_latency_samples(
        result.service_latencies_ns,
        result.ops,
        release_samples,
        "service");
    result.offered_latency_distribution = finalize_latency_samples(
        result.offered_latencies_ns,
        result.ops,
        release_samples,
        "offered");
    result.source_latency_distribution = finalize_latency_samples(
        result.source_latencies_ns,
        result.ops,
        release_samples,
        "source");

    // The three distributions are the canonical per-parent observations.
    // Controller counters accumulate the same waits online, but subtracting
    // large, nearby timestamps per request and then reducing them in a
    // different order leaves a rounding bound that grows with the number of
    // additions. Preserve the online reductions as an independent audit,
    // using the standard gamma_n bound for sequential floating-point sums,
    // then publish the adjacent-distribution residuals so the reported
    // latency partition is exactly self-consistent.
    const auto reduction_tolerance = [count = result.ops](
        double lhs,
        double rhs,
        double reference_total,
        double reference_prefix) {
        const long double epsilon =
            std::numeric_limits<double>::epsilon();
        const long double n_epsilon =
            static_cast<long double>(std::max<std::uint64_t>(count, 1)) *
            epsilon;
        if (!(n_epsilon < 0.5L)) {
            throw std::runtime_error(
                "latency reduction is too large for a bounded double audit");
        }
        const long double gamma_n = n_epsilon / (1.0L - n_epsilon);
        const long double scale = std::max({
            1.0L,
            std::abs(static_cast<long double>(lhs)),
            std::abs(static_cast<long double>(rhs)),
            std::abs(static_cast<long double>(reference_total)),
            std::abs(static_cast<long double>(reference_prefix)),
        });
        // Covers the online sum, both independently reduced distributions,
        // and their element-wise timestamp subtractions. The resulting
        // tolerance remains O(n*epsilon): for this workload it is
        // parts-per-billion, while any missing nanosecond per request is
        // still rejected.
        return static_cast<double>(
            (8.0L * gamma_n + 64.0L * epsilon) * scale);
    };
    const auto canonical_residual = [&reduction_tolerance](
        double total,
        double prefix,
        std::string_view name) {
        double residual = total - prefix;
        const double tolerance =
            reduction_tolerance(residual, 0.0, total, prefix);
        if (residual < 0.0 && std::abs(residual) <= tolerance) {
            residual = 0.0;
        }
        if (!std::isfinite(residual) || residual < 0.0) {
            throw std::runtime_error(
                std::string(name) +
                " latency residual is non-finite or negative");
        }
        return residual;
    };
    const auto audit_online_reduction = [&reduction_tolerance](
        double online,
        double canonical,
        double reference_total,
        double reference_prefix,
        std::string_view name) {
        const double tolerance = reduction_tolerance(
            online,
            canonical,
            reference_total,
            reference_prefix);
        if (!std::isfinite(online) ||
            std::abs(online - canonical) > tolerance) {
            throw std::runtime_error(
                std::string(name) +
                " online wait reduction diverges from latency samples "
                "(difference=" +
                std::to_string(std::abs(online - canonical)) +
                " ns-work, tolerance=" +
                std::to_string(tolerance) + " ns-work)");
        }
    };
    const double canonical_admission_wait = canonical_residual(
        result.offered_latency_distribution->sum_work_ns,
        result.service_latency_distribution->sum_work_ns,
        "front-end admission");
    const double canonical_phase_wait = canonical_residual(
        result.source_latency_distribution->sum_work_ns,
        result.offered_latency_distribution->sum_work_ns,
        "phase dependency");
    audit_online_reduction(
        result.front_end_admission_wait_work_ns,
        canonical_admission_wait,
        result.offered_latency_distribution->sum_work_ns,
        result.service_latency_distribution->sum_work_ns,
        "front-end admission");
    audit_online_reduction(
        result.phase_dependency_wait_work_ns,
        canonical_phase_wait,
        result.source_latency_distribution->sum_work_ns,
        result.offered_latency_distribution->sum_work_ns,
        "phase dependency");
    result.front_end_admission_wait_work_ns =
        canonical_admission_wait;
    result.phase_dependency_wait_work_ns = canonical_phase_wait;
}

const LatencyDistribution& latency_distribution(const ScenarioResult& result) {
    if (!result.offered_latency_distribution) {
        throw std::runtime_error(
            "scenario offered latency distribution was not finalized before "
            "reporting");
    }
    return *result.offered_latency_distribution;
}

const LatencyDistribution& service_latency_distribution(
    const ScenarioResult& result) {
    if (!result.service_latency_distribution) {
        throw std::runtime_error(
            "scenario service latency distribution was not finalized before "
            "reporting");
    }
    return *result.service_latency_distribution;
}

const LatencyDistribution& source_latency_distribution(
    const ScenarioResult& result) {
    if (!result.source_latency_distribution) {
        throw std::runtime_error(
            "scenario source latency distribution was not finalized before "
            "reporting");
    }
    return *result.source_latency_distribution;
}

ResourceBusyMetric resource_busy_metric(
    double busy_ns,
    std::uint64_t resource_count,
    double active_span_ns) {
    if (!std::isfinite(busy_ns) || busy_ns < 0.0 ||
        !std::isfinite(active_span_ns) || active_span_ns < 0.0) {
        throw std::runtime_error("resource busy accounting is non-finite or negative");
    }
    const long double capacity = static_cast<long double>(resource_count) *
        static_cast<long double>(active_span_ns);
    if (capacity > static_cast<long double>(std::numeric_limits<double>::max())) {
        throw std::runtime_error("resource capacity-time exceeds finite double range");
    }
    ResourceBusyMetric metric;
    metric.busy_ns = busy_ns;
    metric.resource_count = resource_count;
    metric.active_span_ns = active_span_ns;
    metric.capacity_time_ns = static_cast<double>(capacity);
    metric.utilization = metric.capacity_time_ns <= 0.0 ? 0.0 :
        metric.busy_ns / metric.capacity_time_ns;
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
        std::max({1.0, metric.busy_ns, metric.capacity_time_ns});
    if (metric.busy_ns > metric.capacity_time_ns + tolerance) {
        throw std::runtime_error(
            "resource busy work exceeds the reported capacity-time");
    }
    return metric;
}

bool time_close(double lhs, double rhs) {
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    // Latency work is reduced along independent paths: per-operation samples
    // are summed once for each distribution, while controller wait counters
    // are accumulated online. At 100M+ operations, their mathematically
    // equivalent totals can differ by hundreds of double ULPs solely from
    // addition order. 4096 ULP remains a sub-parts-per-trillion check and
    // still rejects any material missing stage.
    return std::abs(lhs - rhs) <=
        4096.0 * std::numeric_limits<double>::epsilon() * scale;
}

TimeBreakdownSnapshot make_time_breakdown(const ScenarioResult& result) {
    const auto& offered = latency_distribution(result);
    const auto& service = service_latency_distribution(result);
    const auto& source = source_latency_distribution(result);
    if (offered.count != result.ops ||
        service.count != result.ops ||
        source.count != result.ops) {
        throw std::runtime_error(
            "time breakdown latency counts do not match operation count");
    }
    if (!std::isfinite(result.first_arrival_ns) ||
        !std::isfinite(result.last_arrival_ns) ||
        !std::isfinite(result.user_finish_ns) ||
        !std::isfinite(result.finish_ns) ||
        result.last_arrival_ns < result.first_arrival_ns ||
        result.user_finish_ns < result.last_arrival_ns ||
        result.finish_ns < result.user_finish_ns) {
        throw std::runtime_error(
            "time breakdown wall-clock frontiers violate causal order");
    }
    TimeBreakdownSnapshot out;
    auto& wall = out.wall_clock;
    wall.trace_origin_ns = result.first_arrival_ns;
    wall.last_offered_arrival_ns = result.last_arrival_ns;
    wall.last_user_completion_ns = result.user_finish_ns;
    wall.quiescent_finish_ns = result.finish_ns;
    wall.offered_arrival_span_ns =
        result.last_arrival_ns - result.first_arrival_ns;
    wall.post_offer_user_completion_tail_ns =
        result.user_finish_ns - result.last_arrival_ns;
    wall.user_completion_span_ns =
        result.user_finish_ns - result.first_arrival_ns;
    wall.drain_tail_ns = result.finish_ns - result.user_finish_ns;
    wall.makespan_ns = result.finish_ns - result.first_arrival_ns;
    if (!time_close(
            wall.offered_arrival_span_ns +
                wall.post_offer_user_completion_tail_ns,
            wall.user_completion_span_ns) ||
        !time_close(
            wall.user_completion_span_ns + wall.drain_tail_ns,
            wall.makespan_ns)) {
        throw std::runtime_error(
            "time breakdown wall-clock spans do not conserve makespan");
    }

    auto& latency = out.latency_work;
    latency.user_count = offered.count;
    latency.service_to_user_completion_sum_work_ns =
        service.sum_work_ns;
    latency.phase_barriers = result.phase_barriers;
    latency.phase_dependency_waited_ops =
        result.phase_dependency_waited_ops;
    latency.phase_dependency_wait_work_ns =
        result.phase_dependency_wait_work_ns;
    latency.phase_dependency_max_wait_ns =
        result.phase_dependency_max_wait_ns;
    latency.front_end_admission_waited_ops = result.front_end_admission_waited_ops;
    latency.front_end_admission_wait_work_ns =
        result.front_end_admission_wait_work_ns;
    latency.front_end_admission_max_wait_ns =
        result.front_end_admission_max_wait_ns;
    latency.offered_to_user_completion_sum_work_ns =
        offered.sum_work_ns;
    latency.source_to_user_completion_sum_work_ns =
        source.sum_work_ns;
    latency.average_ns = offered.average_ns;
    latency.p50_ns = offered.p50_ns;
    latency.p95_ns = offered.p95_ns;
    latency.max_ns = offered.max_ns;
    latency.service_average_ns = service.average_ns;
    latency.service_p50_ns = service.p50_ns;
    latency.service_p95_ns = service.p95_ns;
    latency.service_max_ns = service.max_ns;
    latency.source_average_ns = source.average_ns;
    latency.source_p50_ns = source.p50_ns;
    latency.source_p95_ns = source.p95_ns;
    latency.source_max_ns = source.max_ns;

    if (!time_close(
            latency.service_to_user_completion_sum_work_ns +
                latency.front_end_admission_wait_work_ns,
            latency.offered_to_user_completion_sum_work_ns) ||
        !time_close(
            latency.offered_to_user_completion_sum_work_ns +
                latency.phase_dependency_wait_work_ns,
            latency.source_to_user_completion_sum_work_ns)) {
        throw std::runtime_error(
            "latency distributions do not conserve service, front-end, and "
            "phase wait work");
    }

    if (result.has_hbm) {
        out.hbm_stage_work = result.hbm_stats.stage_work;
    }
    if (result.has_hbf) {
        out.hbf_stage_work = result.hbf_stats.stage_work;
    }
    if (result.has_external_backing) {
        out.external_backing_stage_work = result.external_backing_stats.stage_work;
    }
    const auto& link = result.base_die_link_stats;
    out.base_die_link_stage_work = BaseDieLinkStageWorkSnapshot{
        .read_queue_wait_work_ns = link.read_queue_wait_ns,
        .write_queue_wait_work_ns = link.write_queue_wait_ns,
        .read_serialization_work_ns = link.read_busy_ns,
        .write_serialization_work_ns = link.write_busy_ns,
        .read_fixed_latency_work_ns = link.read_fixed_latency_work_ns,
        .write_fixed_latency_work_ns = link.write_fixed_latency_work_ns,
    };
    auto& streaming_work = out.layer_streaming_controller_stage_work;
    if (result.layer_streaming_stats) {
        const auto& stats = *result.layer_streaming_stats;
        streaming_work.present = true;
        streaming_work.backing_admission_wait_work_ns =
            stats.backing_admission_wait_work_ns;
        streaming_work.backing_admission_max_wait_ns =
            stats.backing_admission_max_wait_ns;
        streaming_work.user_wait_work_ns = stats.user_wait_work_ns;
        streaming_work.user_max_wait_ns = stats.user_max_wait_ns;
        streaming_work.exposed_prefetch_ns = stats.exposed_prefetch_ns;
        streaming_work.hidden_prefetch_ns = stats.hidden_prefetch_ns;
        streaming_work.buffer_reuse_wait_work_ns =
            stats.buffer_reuse_wait_work_ns;
    }
    out.cooperative_write_controller_stage_work =
        CooperativeWriteControllerStageWorkSnapshot{
            .present = result.cooperative_write_controller_present,
            .full_waited_ops = result.hbm_write_buffer_full_waits,
            .full_wait_work_ns = result.hbm_write_buffer_wait_ns,
        };

    auto& busy = out.resource_busy;
    if (result.has_hbm) {
        const auto span = result.hbm_stats.active_span_ns();
        busy.hbm_data_bus = resource_busy_metric(
            result.hbm_stats.bus_busy_ns, result.hbm_stats.pseudo_channels, span);
    }
    if (result.has_hbf) {
        const auto& stats = result.hbf_stats;
        const auto span = stats.active_span_ns();
        busy.hbf_logic_ingress = resource_busy_metric(
            stats.logic_ingress_busy_ns, stats.logic_ingress_resources, span);
        busy.hbf_mapping_dram_issue = resource_busy_metric(
            stats.mapping_dram_issue_busy_ns,
            stats.mapping_dram_resources,
            span);
        busy.hbf_plane_media = resource_busy_metric(stats.media_busy_ns, stats.planes, span);
        busy.hbf_media_lane = resource_busy_metric(
            stats.read_lane_busy_ns, stats.media_lanes, span);
        busy.hbf_subarray = resource_busy_metric(
            stats.subarray_read_busy_ns, stats.subarrays, span);
        busy.hbf_page_buffer_bank = resource_busy_metric(
            stats.page_buffer_bank_busy_ns, stats.page_buffer_banks, span);
        busy.hbf_flash_source_queue = resource_busy_metric(
            stats.flash_source_queue_busy_ns,
            stats.flash_source_queue_resources,
            span);
        busy.hbf_channel_command = resource_busy_metric(
            stats.channel_command_busy_ns, stats.channel_command_resources, span);
        busy.hbf_channel_data = resource_busy_metric(
            stats.channel_data_busy_ns, stats.channel_data_resources, span);
        busy.hbf_tsv = resource_busy_metric(
            stats.tsv_busy_ns, stats.tsv_resources, span);
        busy.hbf_sram = resource_busy_metric(
            stats.sram_busy_ns, stats.sram_resources, span);
        busy.hbf_hbio_command = resource_busy_metric(
            stats.hb_io_command_busy_ns, stats.stacks, span);
        busy.hbf_hbio_data = resource_busy_metric(
            stats.hb_io_data_busy_ns, stats.stacks, span);
        busy.hbf_sequencer = resource_busy_metric(
            stats.sequencer_busy_ns, stats.dies, span);
        busy.hbf_ecc_issue = resource_busy_metric(
            stats.ecc_issue_busy_ns, stats.dies, span);
    }
    if (result.has_external_backing) {
        const auto& stats = result.external_backing_stats;
        busy.external_controller = resource_busy_metric(
            stats.controller_issue_busy_ns,
            1,
            stats.controller_active_span_ns());
        busy.external_media = resource_busy_metric(
            stats.media_read_busy_ns + stats.media_write_busy_ns,
            stats.media_channels,
            stats.media_active_span_ns());
        busy.external_link_m2s = resource_busy_metric(
            stats.m2s_busy_ns,
            1,
            stats.m2s_active_span_ns());
        busy.external_link_s2m = resource_busy_metric(
            stats.s2m_busy_ns,
            1,
            stats.s2m_active_span_ns());
    }
    const auto link_span = link.active_span_ns();
    busy.base_die_link_read = resource_busy_metric(
        link.read_busy_ns, link.links, link_span);
    busy.base_die_link_write = resource_busy_metric(
        link.write_busy_ns, link.links, link_span);
    return out;
}

double user_completion_throughput_GBps(const ScenarioResult& result) {
    const auto elapsed_ns = result.user_finish_ns - result.first_arrival_ns;
    return elapsed_ns <= 0.0 ? 0.0 :
        static_cast<double>(result.logical_bytes) / elapsed_ns;
}

double makespan_throughput_GBps(const ScenarioResult& result) {
    const auto elapsed_ns = result.finish_ns - result.first_arrival_ns;
    return elapsed_ns <= 0.0 ? 0.0 :
        static_cast<double>(result.logical_bytes) / elapsed_ns;
}

void print_run_config(
    const Options& options,
    std::size_t op_count,
    std::uint64_t trace_footprint_bytes) {
    const auto hbm = make_hbm_config(options);
    const auto external = make_external_backing_config(options);
    const auto hbf_bytes = hbf_capacity_bytes(options);
    const auto ratio = static_cast<long double>(hbf_bytes) /
        static_cast<long double>(options.hbm_capacity_bytes);
    const auto hbf_codeword_bytes = options.hbf_page_size + options.hbf_oob_bytes;
    const double ecc_decode_ii_ns = hbfsim::physical::transfer_time_ns(
        hbf_codeword_bytes, options.hbf_ecc_decode_raw_bw_GBps_per_die);
    const double ecc_encode_ii_ns = hbfsim::physical::transfer_time_ns(
        hbf_codeword_bytes, options.hbf_ecc_encode_raw_bw_GBps_per_die);
    std::cout << "config:"
              << " ops=" << op_count
              << " line_size=" << options.line_size
              << " trace_unique_line_bytes=" << trace_footprint_bytes
              << " hbm_capacity=" << format_bytes(options.hbm_capacity_bytes)
              << " hbf_capacity=" << format_bytes(hbf_bytes)
              << " hbf_to_hbm=" << fixed(static_cast<double>(ratio), 3) << "x"
              << " flat_hbm_bytes=" << options.flat_hbm_bytes
              << " cooperative_write_region_base="
              << hbm_cooperative_region_base_addr(options)
              << " cooperative_write_region_bytes="
              << options.hbf_hbm_write_buffer_bytes
              << " layer_buffer_bytes=" << options.layer_buffer_bytes
              << " explicit_residency_contract="
              << (options.explicit_residency_contract ? "true" : "false")
              << " behavioral_hbm_bytes="
              << (options.behavioral_hbm_bytes == 0 ?
                  options.hbm_capacity_bytes :
                  options.behavioral_hbm_bytes)
              << " behavioral_promotion_threshold="
              << options.behavioral_promotion_threshold
              << " behavioral_history_pages="
              << options.behavioral_history_pages
              << " max_outstanding_requests="
              << options.max_outstanding_requests
              << " max_hbm_outstanding_requests="
              << options.max_hbm_outstanding_requests
              << " max_hbf_outstanding_requests="
              << options.max_hbf_outstanding_requests
              << " address_heatmap_bins=" << options.address_heatmap_bins
              << " base_die_link_read_bw=" << fixed(options.base_die_link_read_bw_GBps)
              << " base_die_link_write_bw=" << fixed(options.base_die_link_write_bw_GBps)
              << " base_die_link_latency_ns=" << fixed(options.base_die_link_latency_ns)
              << " external_backing_kind="
              << hbfsim::physical::external::to_string(external.kind)
              << " external_backing_capacity="
              << format_bytes(external.capacity_bytes)
              << " external_backing_page_size="
              << external.page_size_bytes
              << " external_backing_media_channels="
              << external.media_channels
              << " external_backing_max_outstanding_requests="
              << external.max_outstanding_requests
              << " external_backing_controller_issue_ns="
              << fixed(external.controller_issue_ns)
              << " external_backing_controller_processing_ns="
              << fixed(external.controller_processing_ns)
              << " external_backing_media_read_latency_ns="
              << fixed(external.media_read_latency_ns)
              << " external_backing_media_write_latency_ns="
              << fixed(external.media_write_latency_ns)
              << " external_backing_media_read_bw="
              << fixed(external.media_read_bandwidth_GBps)
              << " external_backing_media_write_bw="
              << fixed(external.media_write_bandwidth_GBps)
              << " external_backing_m2s_bw="
              << fixed(external.m2s_bandwidth_GBps)
              << " external_backing_s2m_bw="
              << fixed(external.s2m_bandwidth_GBps)
              << " external_backing_one_way_propagation_ns="
              << fixed(external.one_way_propagation_ns)
              << " external_backing_command_bytes="
              << external.command_bytes
              << " external_backing_completion_bytes="
              << external.completion_bytes;
    if (options.generate_llm_path || options.generate_semantic_llm_path) {
        std::cout << " llm_weight_base=0x" << std::hex << options.llm_weight_base
                  << " llm_kv_base=0x" << options.llm_kv_base
                  << " llm_scratch_base=0x" << options.llm_scratch_base << std::dec;
    }
    std::cout << " hbm_channels=" << options.hbm_channels
              << " hbm_pseudo_channels=" << options.hbm_pseudo_channels
              << " hbm_bank_groups_per_pseudo_channel="
              << options.hbm_bank_groups_per_pseudo_channel
              << " hbm_address_mapping_scheme=" << hbm.address_mapping_scheme()
              << " hbm_channel_width_bits=" << hbm.channel_width_bits
              << " hbm_pseudo_channel_width_bits="
              << hbm.pseudo_channel_width_bits()
              << " hbm_pin_rate_Gbps=" << fixed(hbm.pin_rate_Gbps, 3)
              << " hbm_command_clock_MHz=" << fixed(hbm.command_clock_MHz(), 3)
              << " hbm_tck_ns=" << fixed(hbm.command_clock_period_ns(), 6)
              << " hbm_burst_length=" << hbm.burst_length
              << " hbm_burst_bytes=" << hbm.burst_bytes()
              << " hbm_burst_ns=" << fixed(hbm.burst_duration_ns(), 6)
              << " hbm_channel_bw_GBps="
              << fixed(hbm.channel_bandwidth_GBps(), 3)
              << " hbm_stack_peak_bw_GBps="
              << fixed(hbm.channel_bandwidth_GBps() * hbm.channels_per_stack, 3)
              << " hbm_system_peak_bw_GBps="
              << fixed(
                  hbm.channel_bandwidth_GBps() * hbm.channels_per_stack * hbm.stacks,
                  3)
              << " hbm_tccd_s_ns=" << fixed(hbm.tCCD_S_ns(), 6)
              << " hbm_tccd_l_ns=" << fixed(hbm.tCCD_L_ns(), 6)
              << " hbm_refresh=" << (options.hbm_refresh_enabled ? "yes" : "no")
              << " hbf_placement_mapping_scheme="
              << hbfsim::physical::hbf::kPlacementMappingScheme
              << " hbf_stacks=" << options.hbf_stacks
              << " hbf_channels=" << options.hbf_channels
              << " hbf_page_size=" << options.hbf_page_size
              << " hbf_oob_bytes=" << options.hbf_oob_bytes
              << " hbf_media_lanes_per_plane=" << options.hbf_media_lanes_per_plane
              << " hbf_subarrays_per_plane=" << options.hbf_subarrays_per_plane
              << " hbf_page_buffer_banks_per_plane="
              << options.hbf_page_buffer_banks_per_plane
              << " hbf_read_ns=" << fixed(options.hbf_read_ns)
              << " hbf_program_ns=" << fixed(options.hbf_program_ns)
              << " hbf_program_verify_ns=" << fixed(options.hbf_program_verify_ns)
              << " hbf_ecc_decode_latency_ns="
              << fixed(options.hbf_ecc_decode_latency_ns)
              << " hbf_ecc_encode_latency_ns="
              << fixed(options.hbf_ecc_encode_latency_ns)
              << " hbf_ecc_decode_raw_bw_per_die="
              << fixed(options.hbf_ecc_decode_raw_bw_GBps_per_die, 6)
              << " hbf_ecc_encode_raw_bw_per_die="
              << fixed(options.hbf_ecc_encode_raw_bw_GBps_per_die, 6)
              << " hbf_ecc_codeword_size_bytes=" << hbf_codeword_bytes
              << " hbf_ecc_decode_ii_ns=" << fixed(ecc_decode_ii_ns, 6)
              << " hbf_ecc_encode_ii_ns=" << fixed(ecc_encode_ii_ns, 6)
              << " hbf_ecc_issue_topology=shared-per-die"
              << " hbf_media_lane_bw=" << fixed(options.hbf_media_lane_bw_GBps)
              << " hbf_logic_sram_bw=" << fixed(options.hbf_logic_sram_bw_GBps)
              << " hbf_page_buffer_bw=" << fixed(options.hbf_page_buffer_bw_GBps)
              << " hbf_write_coalescing=" << (options.hbf_write_coalescing ? "yes" : "no")
              << " hbf_write_buffer_pages=" << options.hbf_write_buffer_pages
              << " hbf_write_buffer_flush_threshold="
              << options.hbf_write_buffer_flush_threshold_pages
              << " hbf_gc_reserved_blocks_per_plane="
              << options.hbf_gc_reserved_free_blocks_per_plane
              << " hbf_gc_wear_weight=" << fixed(options.hbf_gc_wear_leveling_weight)
              << '\n';
}

// ---- Comparison-table schema -----------------------------------------
// ONE list defines the result table: header, console cell, CSV cell. Both
// surfaces render the same columns in the same order by construction, so
// they cannot drift apart again. Where the two spellings legitimately
// differ today (percent-scaled+"%" on the console vs raw fraction in the
// CSV; "n/a" vs empty for a missing device), the difference is declared
// per column HERE — not implied by two hand-maintained emitters 300 lines
// apart. Adding a column is one entry.
std::string csv_escape(const std::string& value);

struct ResultCell {
    std::string console;
    std::string csv;
};

ResultCell cell_same(std::string value) {
    return {value, value};
}

ResultCell cell_count(std::uint64_t value) {
    return cell_same(std::to_string(value));
}

ResultCell cell_bool(bool value) {
    return cell_same(value ? "true" : "false");
}

ResultCell cell_fixed(double value) {
    return cell_same(fixed(value));
}

ResultCell cell_percent(double fraction) {
    return {fixed(fraction * 100.0) + "%", fixed(fraction)};
}

ResultCell cell_missing() {
    return {"n/a", ""};
}

ResultCell cell_optional_fixed(std::optional<double> value, int precision = 3) {
    return value ? cell_same(fixed(*value, precision)) : cell_missing();
}

ResultCell cell_resource_utilization(
    double busy_ns,
    std::uint64_t resource_count,
    double active_span_ns) {
    return cell_percent(
        resource_busy_metric(busy_ns, resource_count, active_span_ns).utilization);
}

struct ResultColumn {
    const char* header;
    std::function<ResultCell(const ScenarioResult&)> cell;
};

// Gate a cell on device presence (console "n/a", CSV empty when absent).
template <typename Fn>
std::function<ResultCell(const ScenarioResult&)> hbm_cell(Fn fn) {
    return [fn](const ScenarioResult& r) {
        return r.has_hbm ? fn(r) : cell_missing();
    };
}

template <typename Fn>
std::function<ResultCell(const ScenarioResult&)> hbf_cell(Fn fn) {
    return [fn](const ScenarioResult& r) {
        return r.has_hbf ? fn(r) : cell_missing();
    };
}

template <typename Fn>
std::function<ResultCell(const ScenarioResult&)> external_backing_cell(Fn fn) {
    return [fn](const ScenarioResult& r) {
        return r.has_external_backing ? fn(r) : cell_missing();
    };
}

template <typename Fn>
std::function<ResultCell(const ScenarioResult&)> layer_streaming_cell(Fn fn) {
    return [fn](const ScenarioResult& r) {
        return r.layer_streaming_stats ?
            fn(*r.layer_streaming_stats) : cell_missing();
    };
}

template <typename Fn>
std::function<ResultCell(const ScenarioResult&)> behavioral_tiering_cell(
    Fn fn) {
    return [fn](const ScenarioResult& r) {
        return r.behavioral_tiering_stats ?
            fn(*r.behavioral_tiering_stats) : cell_missing();
    };
}

std::string ratio_text(std::uint64_t active, std::uint64_t total) {
    return std::to_string(active) + "/" + std::to_string(total);
}

const char* layer_streaming_mode_name(const LayerStreamingStats& stats) {
    (void)stats;
    return "layer_streaming";
}

const char* layer_streaming_residency_policy_name(
    const LayerStreamingStats& stats) {
    return stats.explicit_residency_contract ?
        "capacity-aware-static-weight-prefix-v2" :
        "trace-derived-first-touch";
}

const std::vector<ResultColumn>& result_columns() {
    using R = const ScenarioResult&;
    static const std::vector<ResultColumn> columns = {
        {"scenario", [](R r) { return ResultCell{r.name, csv_escape(r.name)}; }},
        {"summary_schema_version", [](R) { return cell_count(16); }},
        {"ops", [](R r) { return cell_count(r.ops); }},
        {"reads", [](R r) { return cell_count(r.reads); }},
        {"writes", [](R r) { return cell_count(r.writes); }},
        {"hbm_accesses", [](R r) { return cell_count(r.hbm_accesses); }},
        {"hbf_accesses", [](R r) { return cell_count(r.hbf_accesses); }},
        {"external_accesses", [](R r) { return cell_count(r.external_accesses); }},
        {"hbm_user_accesses", [](R r) { return cell_count(r.hbm_user_accesses); }},
        {"hbf_user_accesses", [](R r) { return cell_count(r.hbf_user_accesses); }},
        {"hbf_direct_user_ops", [](R r) { return cell_count(r.hbf_direct_user_ops); }},
        {"hbm_background_accesses", [](R r) { return cell_count(r.hbm_background_accesses); }},
        {"hbf_background_accesses", [](R r) { return cell_count(r.hbf_background_accesses); }},
        {"external_background_accesses", [](R r) {
            return cell_count(r.external_background_accesses);
        }},
        {"behavioral_policy", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_same(
                hbfsim::physical::hybrid::to_string(
                    s.admission_policy));
        })},
        {"behavioral_semantic_inputs_consumed", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_same(
                s.semantic_inputs_consumed ? "true" : "false");
        })},
        {"behavioral_hbm_tier_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbm_tier_pages);
        })},
        {"behavioral_hbm_tier_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbm_tier_bytes);
        })},
        {"behavioral_promotion_threshold", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.promotion_threshold);
        })},
        {"behavioral_history_capacity_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.history_capacity_pages);
        })},
        {"behavioral_page_observations", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.page_observations);
        })},
        {"behavioral_hbm_hits", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbm_hits);
        })},
        {"behavioral_hbf_bypasses", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbf_bypasses);
        })},
        {"behavioral_cold_misses", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.cold_misses);
        })},
        {"behavioral_history_hits", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.history_hits);
        })},
        {"behavioral_promotions", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.promotions);
        })},
        {"behavioral_promotions_with_backing_fill",
         behavioral_tiering_cell([](const BehavioralTieringStats& s) {
            return cell_count(s.promotions_with_backing_fill);
        })},
        {"behavioral_promotions_without_backing_fill",
         behavioral_tiering_cell([](const BehavioralTieringStats& s) {
            return cell_count(s.promotions_without_backing_fill);
        })},
        {"behavioral_clean_evictions", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.clean_evictions);
        })},
        {"behavioral_dirty_evictions", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.dirty_evictions);
        })},
        {"behavioral_history_evictions", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.history_evictions);
        })},
        {"behavioral_hbm_foreground_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbm_foreground_bytes);
        })},
        {"behavioral_hbf_bypass_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbf_bypass_bytes);
        })},
        {"behavioral_backing_fill_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.backing_fill_pages);
        })},
        {"behavioral_backing_fill_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.backing_fill_bytes);
        })},
        {"behavioral_hbm_install_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbm_install_pages);
        })},
        {"behavioral_hbm_install_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbm_install_bytes);
        })},
        {"behavioral_dirty_writeback_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.dirty_writeback_pages);
        })},
        {"behavioral_dirty_writeback_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.dirty_writeback_bytes);
        })},
        {"behavioral_drain_writeback_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.drain_writeback_pages);
        })},
        {"behavioral_drain_writeback_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.drain_writeback_bytes);
        })},
        {"behavioral_peak_resident_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.peak_resident_pages);
        })},
        {"behavioral_final_resident_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.final_resident_pages);
        })},
        {"behavioral_final_dirty_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.final_dirty_pages);
        })},
        {"behavioral_peak_history_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.peak_history_pages);
        })},
        {"behavioral_peak_installing_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.peak_installing_pages);
        })},
        {"behavioral_capacity_stalled_promotions",
         behavioral_tiering_cell([](const BehavioralTieringStats& s) {
            return cell_count(s.capacity_stalled_promotions);
        })},
        {"behavioral_capacity_stall_work_ns", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_fixed(s.capacity_stall_work_ns);
        })},
        {"behavioral_transition_waited_transactions",
         behavioral_tiering_cell([](const BehavioralTieringStats& s) {
            return cell_count(s.transition_waited_transactions);
        })},
        {"behavioral_transition_wait_work_ns", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_fixed(s.transition_wait_work_ns);
        })},
        {"behavioral_transition_max_wait_ns", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_fixed(s.transition_max_wait_ns);
        })},
        {"behavioral_decision_fingerprint", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.decision_fingerprint);
        })},
        {"layer_streaming_mode", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_same(layer_streaming_mode_name(s));
        })},
        {"layer_streaming_backing", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_same(hbfsim::physical::hybrid::to_string(s.backing));
        })},
        {"layer_streaming_residency_policy", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_same(layer_streaming_residency_policy_name(s));
        })},
        {"layer_streaming_compact_resident_mapping", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_same(s.compact_resident_mapping ? "true" : "false");
        })},
        {"layer_streaming_semantic_inputs_consumed", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_same(s.semantic_inputs_consumed ? "true" : "false");
        })},
        {"layer_streaming_explicit_residency_contract", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_same(s.explicit_residency_contract ? "true" : "false");
        })},
        {"layer_streaming_address_footprint_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.address_footprint_bytes);
        })},
        {"layer_streaming_unique_resident_footprint_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.unique_resident_footprint_pages);
        })},
        {"layer_streaming_unique_resident_footprint_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.unique_resident_footprint_bytes);
        })},
        {"layer_streaming_capacity_pressure_basis_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.capacity_pressure_basis_bytes);
        })},
        {"layer_streaming_footprint_page_rounding_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.footprint_page_rounding_bytes);
        })},
        {"layer_streaming_hbm_capacity_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hbm_capacity_bytes);
        })},
        {"layer_streaming_hbm_capacity_pressure", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.hbm_capacity_pressure);
        })},
        {"layer_streaming_layers", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.layers);
        })},
        {"layer_streaming_explicit_layer_requests", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.explicit_layer_requests);
        })},
        {"layer_streaming_explicit_compute_layers", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.explicit_compute_layers);
        })},
        {"layer_streaming_compute_work_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.compute_work_ns);
        })},
        {"layer_streaming_hbm_only_resident_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hbm_only_resident_pages);
        })},
        {"layer_streaming_hbm_only_resident_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hbm_only_resident_bytes);
        })},
        {"layer_streaming_hot_kv_candidate_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hot_kv_candidate_pages);
        })},
        {"layer_streaming_hot_kv_candidate_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hot_kv_candidate_bytes);
        })},
        {"layer_streaming_hot_kv_resident_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hot_kv_resident_pages);
        })},
        {"layer_streaming_hot_kv_resident_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hot_kv_resident_bytes);
        })},
        {"layer_streaming_data_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.data_pages);
        })},
        {"layer_streaming_data_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.data_bytes);
        })},
        {"layer_streaming_model_weight_resident_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.model_weight_resident_pages);
        })},
        {"layer_streaming_model_weight_resident_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.model_weight_resident_bytes);
        })},
        {"layer_streaming_model_weight_backing_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.model_weight_backing_pages);
        })},
        {"layer_streaming_model_weight_backing_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.model_weight_backing_bytes);
        })},
        {"layer_streaming_cold_kv_backing_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.cold_kv_backing_pages);
        })},
        {"layer_streaming_cold_kv_backing_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.cold_kv_backing_bytes);
        })},
        {"layer_streaming_unknown_backing_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.unknown_backing_pages);
        })},
        {"layer_streaming_unknown_backing_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.unknown_backing_bytes);
        })},
        {"layer_streaming_backing_unique_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.backing_unique_pages);
        })},
        {"layer_streaming_backing_unique_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.backing_unique_bytes);
        })},
        {"layer_streaming_resident_physical_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.resident_physical_pages);
        })},
        {"layer_streaming_resident_physical_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.resident_physical_bytes);
        })},
        {"layer_streaming_effective_layer_buffer_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.effective_layer_buffer_pages);
        })},
        {"layer_streaming_effective_layer_buffer_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.effective_layer_buffer_bytes);
        })},
        {"layer_streaming_unused_hbm_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.unused_hbm_pages);
        })},
        {"layer_streaming_unused_hbm_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.unused_hbm_bytes);
        })},
        {"layer_streaming_streamed_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.streamed_pages);
        })},
        {"layer_streaming_streamed_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.streamed_bytes);
        })},
        {"layer_streaming_foreground_resident_page_accesses", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.foreground_resident_page_accesses);
        })},
        {"layer_streaming_foreground_buffer_page_accesses", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.foreground_buffer_page_accesses);
        })},
        {"layer_streaming_dirty_pages_written_back", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.dirty_pages_written_back);
        })},
        {"layer_streaming_writeback_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.writeback_bytes);
        })},
        {"layer_streaming_max_layer_data_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.max_layer_data_pages);
        })},
        {"layer_streaming_max_layer_data_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.max_layer_data_bytes);
        })},
        {"layer_streaming_immutable_weight_logical_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.immutable_weight_logical_bytes);
        })},
        {"layer_streaming_runtime_overhead_logical_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.runtime_overhead_logical_bytes);
        })},
        {"layer_streaming_block_table_logical_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.block_table_logical_bytes);
        })},
        {"layer_streaming_active_buffer_logical_bytes_per_slot", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.active_buffer_logical_bytes_per_slot);
        })},
        {"layer_streaming_residency_page_size_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.residency_page_size_bytes);
        })},
        {"layer_streaming_kv_block_stride_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.kv_block_stride_bytes);
        })},
        {"layer_streaming_logical_kv_blocks", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.logical_kv_blocks);
        })},
        {"layer_streaming_hot_kv_blocks", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hot_kv_blocks);
        })},
        {"layer_streaming_cold_kv_blocks", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.cold_kv_blocks);
        })},
        {"layer_streaming_backing_request_credit_limit", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.backing_request_credit_limit);
        })},
        {"layer_streaming_backing_max_inflight_requests", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.backing_max_inflight_requests);
        })},
        {"layer_streaming_backing_admission_waited_requests", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.backing_admission_waited_requests);
        })},
        {"layer_streaming_backing_admission_wait_work_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.backing_admission_wait_work_ns);
        })},
        {"layer_streaming_backing_admission_max_wait_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.backing_admission_max_wait_ns);
        })},
        {"layer_streaming_user_waited_ops", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.user_waited_ops);
        })},
        {"layer_streaming_user_wait_work_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.user_wait_work_ns);
        })},
        {"layer_streaming_user_max_wait_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.user_max_wait_ns);
        })},
        {"layer_streaming_exposed_prefetch_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.exposed_prefetch_ns);
        })},
        {"layer_streaming_hidden_prefetch_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.hidden_prefetch_ns);
        })},
        {"layer_streaming_buffer_reuse_wait_work_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.buffer_reuse_wait_work_ns);
        })},
        {"background_hbf_writes", [](R r) { return cell_count(r.background_hbf_writes); }},
        {"hbf_static_read_bytes", [](R r) { return cell_count(r.hbf_static_read_bytes); }},
        {"hbf_logical_read_bytes", [](R r) { return cell_count(r.hbf_logical_read_bytes); }},
        {"hbf_backing_write_bytes", [](R r) { return cell_count(r.hbf_backing_write_bytes); }},
        {"external_backing_read_bytes", [](R r) {
            return cell_count(r.external_backing_read_bytes);
        }},
        {"external_backing_write_bytes", [](R r) {
            return cell_count(r.external_backing_write_bytes);
        }},
        {"hbm_foreground_bytes", [](R r) { return cell_count(r.hbm_foreground_bytes); }},
        {"hbm_streaming_write_bytes", [](R r) { return cell_count(r.hbm_streaming_write_bytes); }},
        {"base_die_link_read_bytes", [](R r) {
            return cell_count(r.base_die_link_stats.read_bytes);
        }},
        {"base_die_link_write_bytes", [](R r) {
            return cell_count(r.base_die_link_stats.write_bytes);
        }},
        {"base_die_link_read_queue_wait_work_ns", [](R r) {
            return cell_fixed(r.base_die_link_stats.read_queue_wait_ns);
        }},
        {"base_die_link_write_queue_wait_work_ns", [](R r) {
            return cell_fixed(r.base_die_link_stats.write_queue_wait_ns);
        }},
        {"base_die_link_read_serialization_work_ns", [](R r) {
            return cell_fixed(r.base_die_link_stats.read_busy_ns);
        }},
        {"base_die_link_write_serialization_work_ns", [](R r) {
            return cell_fixed(r.base_die_link_stats.write_busy_ns);
        }},
        {"base_die_link_read_fixed_latency_work_ns", [](R r) {
            return cell_fixed(r.base_die_link_stats.read_fixed_latency_work_ns);
        }},
        {"base_die_link_write_fixed_latency_work_ns", [](R r) {
            return cell_fixed(r.base_die_link_stats.write_fixed_latency_work_ns);
        }},
        {"external_backing_kind", external_backing_cell([](R r) {
            return cell_same(
                hbfsim::physical::external::to_string(
                    r.external_backing_stats.kind));
        })},
        {"external_read_requests", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.read_requests);
        })},
        {"external_write_requests", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.write_requests);
        })},
        {"external_read_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.read_bytes);
        })},
        {"external_write_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.write_bytes);
        })},
        {"external_outstanding_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.outstanding_wait_ns);
        })},
        {"external_controller_queue_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.controller_queue_wait_ns);
        })},
        {"external_controller_processing_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.controller_processing_work_ns);
        })},
        {"external_media_queue_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.media_queue_wait_ns);
        })},
        {"external_link_m2s_queue_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.m2s_queue_wait_ns);
        })},
        {"external_link_s2m_queue_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.s2m_queue_wait_ns);
        })},
        {"external_media_read_latency_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.media_read_latency_work_ns);
        })},
        {"external_media_write_latency_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.media_write_latency_work_ns);
        })},
        {"external_transport_propagation_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.transport_propagation_work_ns);
        })},
        {"external_m2s_payload_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.m2s_payload_bytes);
        })},
        {"external_m2s_protocol_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.m2s_protocol_bytes);
        })},
        {"external_m2s_wire_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.m2s_wire_bytes);
        })},
        {"external_s2m_payload_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.s2m_payload_bytes);
        })},
        {"external_s2m_protocol_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.s2m_protocol_bytes);
        })},
        {"external_s2m_wire_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.s2m_wire_bytes);
        })},
        {"external_media_busy_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.media_read_busy_ns +
                r.external_backing_stats.media_write_busy_ns);
        })},
        {"external_media_read_busy_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.media_read_busy_ns);
        })},
        {"external_media_write_busy_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.media_write_busy_ns);
        })},
        {"external_media_resource_count", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.media_channels);
        })},
        {"external_media_active_span_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.media_active_span_ns());
        })},
        {"external_media_capacity_time_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.media_active_span_ns() *
                static_cast<double>(
                    r.external_backing_stats.media_channels));
        })},
        {"external_media_utilization", external_backing_cell([](R r) {
            return cell_percent(r.external_backing_stats.media_utilization());
        })},
        {"external_controller_busy_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.controller_issue_busy_ns);
        })},
        {"external_controller_resource_count", external_backing_cell([](R) {
            return cell_count(1);
        })},
        {"external_controller_active_span_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.controller_active_span_ns());
        })},
        {"external_controller_capacity_time_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.controller_active_span_ns());
        })},
        {"external_controller_utilization", external_backing_cell([](R r) {
            return cell_percent(
                r.external_backing_stats.controller_utilization());
        })},
        {"external_link_m2s_busy_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.m2s_busy_ns);
        })},
        {"external_link_m2s_resource_count", external_backing_cell([](R) {
            return cell_count(1);
        })},
        {"external_link_m2s_active_span_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.m2s_active_span_ns());
        })},
        {"external_link_m2s_capacity_time_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.m2s_active_span_ns());
        })},
        {"external_link_m2s_utilization", external_backing_cell([](R r) {
            return cell_percent(r.external_backing_stats.m2s_utilization());
        })},
        {"external_link_s2m_busy_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.s2m_busy_ns);
        })},
        {"external_link_s2m_resource_count", external_backing_cell([](R) {
            return cell_count(1);
        })},
        {"external_link_s2m_active_span_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.s2m_active_span_ns());
        })},
        {"external_link_s2m_capacity_time_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.s2m_active_span_ns());
        })},
        {"external_link_s2m_utilization", external_backing_cell([](R r) {
            return cell_percent(r.external_backing_stats.s2m_utilization());
        })},
        {"external_backing_ingress_queue_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.stage_work.ingress_queue_wait_ns);
        })},
        {"external_backing_scheduler_queue_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.stage_work.scheduler_queue_wait_ns);
        })},
        {"external_backing_command_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.stage_work.command_ns);
        })},
        {"external_backing_array_read_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.stage_work.array_read_ns);
        })},
        {"external_backing_array_program_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.stage_work.array_program_ns);
        })},
        {"external_backing_channel_transfer_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.stage_work.channel_transfer_ns);
        })},
        {"external_backing_hb_io_transfer_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.stage_work.hb_io_transfer_ns);
        })},
        {"external_backing_transport_latency_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.stage_work.transport_latency_ns);
        })},
        {"external_backing_total_overlapping_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.stage_work.total_work_ns());
        })},
        {"hbm_write_buffer_user_write_bytes", [](R r) {
            return cell_count(r.hbm_write_buffer_user_write_bytes);
        }},
        {"hbm_write_buffer_destaged_bytes", [](R r) {
            return cell_count(r.hbm_write_buffer_destaged_bytes);
        }},
        {"hbm_write_buffer_peak_bytes", [](R r) {
            return cell_count(r.hbm_write_buffer_peak_bytes);
        }},
        {"hbm_write_buffer_full_waits", [](R r) {
            return cell_count(r.hbm_write_buffer_full_waits);
        }},
        {"cooperative_write_controller_full_waited_ops", [](R r) {
            return r.cooperative_write_controller_present ?
                cell_count(r.hbm_write_buffer_full_waits) : cell_missing();
        }},
        {"cooperative_write_controller_full_wait_work_ns", [](R r) {
            return r.cooperative_write_controller_present ?
                cell_fixed(r.hbm_write_buffer_wait_ns) : cell_missing();
        }},
        {"hbf_end_to_end_media_bytes_per_cooperative_user_write_byte", [](R r) {
            if (!r.has_hbf || r.hbm_write_buffer_user_write_bytes == 0) {
                return cell_missing();
            }
            return cell_same(fixed(
                static_cast<double>(r.hbf_stats.physical_write_bytes) /
                static_cast<double>(r.hbm_write_buffer_user_write_bytes),
                6));
        }},
        {"base_die_link_read_busy_ns", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_fixed(r.base_die_link_stats.read_busy_ns);
        }},
        {"base_die_link_read_resource_count", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_count(r.base_die_link_stats.links);
        }},
        {"base_die_link_read_active_span_ns", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_fixed(r.base_die_link_stats.active_span_ns());
        }},
        {"base_die_link_read_capacity_time_ns", [](R r) {
            if (r.base_die_link_stats.links == 0) return cell_missing();
            return cell_fixed(resource_busy_metric(
                r.base_die_link_stats.read_busy_ns,
                r.base_die_link_stats.links,
                r.base_die_link_stats.active_span_ns()).capacity_time_ns);
        }},
        {"base_die_link_read_utilization", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_resource_utilization(
                    r.base_die_link_stats.read_busy_ns,
                    r.base_die_link_stats.links,
                    r.base_die_link_stats.active_span_ns());
        }},
        {"base_die_link_write_busy_ns", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_fixed(r.base_die_link_stats.write_busy_ns);
        }},
        {"base_die_link_write_resource_count", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_count(r.base_die_link_stats.links);
        }},
        {"base_die_link_write_active_span_ns", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_fixed(r.base_die_link_stats.active_span_ns());
        }},
        {"base_die_link_write_capacity_time_ns", [](R r) {
            if (r.base_die_link_stats.links == 0) return cell_missing();
            return cell_fixed(resource_busy_metric(
                r.base_die_link_stats.write_busy_ns,
                r.base_die_link_stats.links,
                r.base_die_link_stats.active_span_ns()).capacity_time_ns);
        }},
        {"base_die_link_write_utilization", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_resource_utilization(
                    r.base_die_link_stats.write_busy_ns,
                    r.base_die_link_stats.links,
                    r.base_die_link_stats.active_span_ns());
        }},
        {"hbm_row_hit_rate", hbm_cell([](R r) { return cell_percent(r.hbm_stats.row_hit_rate()); })},
        {"hbm_address_mapping_work_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.stage_work.address_mapping_ns);
        })},
        {"hbm_scheduler_queue_wait_work_ns", hbm_cell([](R r) {
            return cell_same(fixed(r.hbm_stats.stage_work.scheduler_queue_wait_ns));
        })},
        {"hbm_refresh_stall_work_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.stage_work.refresh_stall_ns);
        })},
        {"hbm_precharge_work_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.stage_work.precharge_ns);
        })},
        {"hbm_activation_work_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.stage_work.activation_ns);
        })},
        {"hbm_command_work_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.stage_work.command_ns);
        })},
        {"hbm_channel_transfer_work_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.stage_work.channel_transfer_ns);
        })},
        {"hbm_total_overlapping_work_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.stage_work.total_work_ns());
        })},
        {"hbm_data_bus_busy_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.bus_busy_ns);
        })},
        {"hbm_data_bus_resource_count", hbm_cell([](R r) {
            return cell_count(r.hbm_stats.pseudo_channels);
        })},
        {"hbm_data_bus_active_span_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.active_span_ns());
        })},
        {"hbm_data_bus_capacity_time_ns", hbm_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbm_stats.bus_busy_ns,
                r.hbm_stats.pseudo_channels,
                r.hbm_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbm_data_bus_utilization", hbm_cell([](R r) {
            return cell_resource_utilization(
                r.hbm_stats.bus_busy_ns,
                r.hbm_stats.pseudo_channels,
                r.hbm_stats.active_span_ns());
        })},
        {"hbm_max_queue_occupancy", hbm_cell([](R r) {
            return cell_count(r.hbm_stats.max_queue_occupancy);
        })},
        {"hbm_bus_parallelism", hbm_cell([](R r) { return cell_same(fixed(r.hbm_stats.bus_parallelism(), 2)); })},
        {"hbm_active_pch", hbm_cell([](R r) {
            return cell_same(ratio_text(r.hbm_stats.active_pseudo_channels, r.hbm_stats.pseudo_channels)); })},
        {"hbm_pch_busy_skew", hbm_cell([](R r) { return cell_same(fixed(r.hbm_stats.pseudo_channel_busy_skew(), 2)); })},
        {"hbf_waf", hbf_cell([](R r) {
            return cell_optional_fixed(r.hbf_stats.waf()); })},
        {"hbf_data_program_payload_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.data_program_payload_bytes); })},
        {"hbf_mapping_program_payload_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_program_payload_bytes); })},
        {"hbf_gc_relocation_payload_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_relocation_payload_bytes); })},
        {"hbf_page_reads", hbf_cell([](R r) { return cell_count(r.hbf_stats.page_reads); })},
        {"hbf_data_programs", hbf_cell([](R r) { return cell_count(r.hbf_stats.data_programs); })},
        {"hbf_page_programs", hbf_cell([](R r) { return cell_count(r.hbf_stats.page_programs); })},
        {"hbf_page_read_admission_events", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.page_read_admission_events);
        })},
        {"hbf_page_read_admission_waited_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.page_read_admission_waited_pages);
        })},
        {"hbf_page_read_admission_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.page_read_admission_wait_ns);
        })},
        {"hbf_page_read_admission_max_wait_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.page_read_admission_max_wait_ns);
        })},
        {"hbf_mapping_page_programs", hbf_cell([](R r) { return cell_count(r.hbf_stats.mapping_page_programs); })},
        {"hbf_resident_mapping_table_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.resident_mapping_table_bytes); })},
        {"hbf_resident_mapping_table_bytes_per_stack", hbf_cell([](R r) {
            return cell_count(
                r.hbf_stats.resident_mapping_table_bytes_per_stack); })},
        {"hbf_resident_mapping_pages_per_stack", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.resident_mapping_pages_per_stack); })},
        {"hbf_mapping_lookup_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_lookup_ops); })},
        {"hbf_mapping_user_lookup_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_user_lookup_ops); })},
        {"hbf_mapping_gc_lookup_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_gc_lookup_ops); })},
        {"hbf_mapping_update_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_update_ops); })},
        {"hbf_mapping_user_update_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_user_update_ops); })},
        {"hbf_mapping_gc_update_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_gc_update_ops); })},
        {"hbf_mapping_dram_wait_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_dram_wait_ops); })},
        {"hbf_mapping_dram_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.mapping_dram_wait_ns); })},
        {"hbf_mapping_dram_wait_max_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.mapping_dram_wait_max_ns); })},
        {"hbf_mapping_dram_issue_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.mapping_dram_issue_busy_ns); })},
        {"hbf_mapping_dram_resources", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_dram_resources); })},
        {"hbf_read_requests", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.read_requests); })},
        {"hbf_program_requests", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.program_requests); })},
        {"hbf_erase_requests", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.erase_requests); })},
        {"hbf_invalidations", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.invalidations); })},
        {"hbf_total_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.total_pages); })},
        {"hbf_free_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.free_pages); })},
        {"hbf_valid_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.valid_pages); })},
        {"hbf_invalid_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.invalid_pages); })},
        {"hbf_pending_program_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.pending_program_pages); })},
        {"hbf_pending_mapping_publications", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.pending_mapping_publications); })},
        {"hbf_static_reserved_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.static_reserved_pages); })},
        {"hbf_initial_logical_data_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.initial_logical_data_pages); })},
        {"hbf_initial_mapping_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.initial_mapping_pages); })},
        {"hbf_compact_initial_logical_data_pages", hbf_cell([](R r) {
            return cell_count(
                r.hbf_stats.compact_initial_logical_data_pages); })},
        {"hbf_compact_initial_mapping_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.compact_initial_mapping_pages); })},
        {"hbf_compact_live_logical_data_pages", hbf_cell([](R r) {
            return cell_count(
                r.hbf_stats.compact_live_logical_data_pages); })},
        {"hbf_compact_live_mapping_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.compact_live_mapping_pages); })},
        {"hbf_compact_retired_logical_data_pages", hbf_cell([](R r) {
            return cell_count(
                r.hbf_stats.compact_retired_logical_data_pages); })},
        {"hbf_compact_retired_mapping_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.compact_retired_mapping_pages); })},
        {"hbf_static_unmaterialized_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.static_unmaterialized_pages); })},
        {"hbf_accounting_verified", hbf_cell([](R r) {
            return cell_bool(r.hbf_stats.accounting_verified); })},
        {"hbf_block_erases", hbf_cell([](R r) { return cell_count(r.hbf_stats.block_erases); })},
        {"hbf_gc_runs", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_runs); })},
        {"hbf_gc_relocations", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_relocations); })},
        {"hbf_gc_data_relocations", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_data_relocations); })},
        {"hbf_gc_mapping_relocations", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_mapping_relocations); })},
        {"hbf_gc_reclaimed_invalid_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_reclaimed_invalid_pages); })},
        {"hbf_gc_user_blocked_runs", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_user_blocked_runs); })},
        {"hbf_flash_scheduler_enqueues", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.flash_scheduler_enqueues); })},
        {"hbf_flash_scheduler_issues", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.flash_scheduler_issues); })},
        {"hbf_ingress_queue_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.ingress_queue_wait_ns);
        })},
        {"hbf_scheduler_queue_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.scheduler_queue_wait_ns);
        })},
        {"hbf_ecc_queue_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.ecc_queue_wait_ns);
        })},
        {"hbf_address_mapping_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.address_mapping_ns);
        })},
        {"hbf_translation_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.translation_ns);
        })},
        {"hbf_mapping_dram_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.mapping_dram_ns);
        })},
        {"hbf_command_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.command_ns);
        })},
        {"hbf_ecc_decode_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.ecc_decode_ops);
        })},
        {"hbf_ecc_encode_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.ecc_encode_ops);
        })},
        {"hbf_ecc_decode_queue_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_decode_queue_wait_ns);
        })},
        {"hbf_ecc_encode_queue_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_encode_queue_wait_ns);
        })},
        {"hbf_ecc_decode_response_latency_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_decode_latency_work_ns);
        })},
        {"hbf_ecc_encode_response_latency_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_encode_latency_work_ns);
        })},
        {"hbf_ecc_decode_issue_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_decode_issue_busy_ns);
        })},
        {"hbf_ecc_encode_issue_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_encode_issue_busy_ns);
        })},
        {"hbf_ecc_decode_codeword_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.ecc_decode_codeword_bytes);
        })},
        {"hbf_ecc_encode_codeword_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.ecc_encode_codeword_bytes);
        })},
        {"hbf_ecc_codeword_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.ecc_codeword_bytes);
        })},
        {"hbf_ecc_issue_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_issue_busy_ns);
        })},
        {"hbf_ecc_issue_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.dies);
        })},
        {"hbf_ecc_issue_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_ecc_issue_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.ecc_issue_busy_ns,
                r.hbf_stats.dies,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_ecc_issue_utilization", hbf_cell([](R r) {
            return cell_percent(r.hbf_stats.ecc_issue_utilization());
        })},
        {"hbf_ecc_issue_parallelism", hbf_cell([](R r) {
            return cell_same(fixed(r.hbf_stats.ecc_issue_parallelism(), 2));
        })},
        {"hbf_ecc_max_inflight_per_die", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.max_ecc_inflight_per_die);
        })},
        {"hbf_write_buffer_slot_waited_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.write_buffer_slot_wait_ops);
        })},
        {"hbf_write_buffer_slot_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.write_buffer_slot_wait_ns);
        })},
        {"hbf_array_read_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.array_read_ns);
        })},
        {"hbf_array_program_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.array_program_ns);
        })},
        {"hbf_program_verify_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.program_verify_ns);
        })},
        {"hbf_array_erase_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.array_erase_ns);
        })},
        {"hbf_media_lane_transfer_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.media_lane_transfer_ns);
        })},
        {"hbf_page_buffer_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.page_buffer_ns);
        })},
        {"hbf_sram_staging_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.sram_staging_ns);
        })},
        {"hbf_channel_transfer_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.channel_transfer_ns);
        })},
        {"hbf_tsv_transfer_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.tsv_transfer_ns);
        })},
        {"hbf_hb_io_transfer_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.hb_io_transfer_ns);
        })},
        {"hbf_transport_latency_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.transport_latency_ns);
        })},
        {"hbf_ecc_response_latency_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.ecc_latency_ns);
        })},
        {"hbf_maintenance_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.maintenance_ns);
        })},
        {"hbf_total_overlapping_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.total_work_ns());
        })},
        {"hbf_logic_ingress_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.logic_ingress_busy_ns);
        })},
        {"hbf_logic_ingress_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.logic_ingress_resources);
        })},
        {"hbf_logic_ingress_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_logic_ingress_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.logic_ingress_busy_ns,
                r.hbf_stats.logic_ingress_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_logic_ingress_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.logic_ingress_busy_ns,
                r.hbf_stats.logic_ingress_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_mapping_dram_issue_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.mapping_dram_issue_busy_ns);
        })},
        {"hbf_mapping_dram_issue_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_dram_resources);
        })},
        {"hbf_mapping_dram_issue_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_mapping_dram_issue_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.mapping_dram_issue_busy_ns,
                r.hbf_stats.mapping_dram_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_mapping_dram_issue_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.mapping_dram_issue_busy_ns,
                r.hbf_stats.mapping_dram_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_plane_media_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.media_busy_ns);
        })},
        {"hbf_plane_media_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.planes);
        })},
        {"hbf_plane_media_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_plane_media_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.media_busy_ns,
                r.hbf_stats.planes,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_plane_media_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.media_busy_ns,
                r.hbf_stats.planes,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_media_lane_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.read_lane_busy_ns);
        })},
        {"hbf_media_lane_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.media_lanes);
        })},
        {"hbf_media_lane_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_media_lane_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.read_lane_busy_ns,
                r.hbf_stats.media_lanes,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_media_lane_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.read_lane_busy_ns,
                r.hbf_stats.media_lanes,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_subarray_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.subarray_read_busy_ns);
        })},
        {"hbf_subarray_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.subarrays);
        })},
        {"hbf_subarray_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_subarray_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.subarray_read_busy_ns,
                r.hbf_stats.subarrays,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_subarray_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.subarray_read_busy_ns,
                r.hbf_stats.subarrays,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_page_buffer_bank_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.page_buffer_bank_busy_ns);
        })},
        {"hbf_page_buffer_bank_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.page_buffer_banks);
        })},
        {"hbf_page_buffer_bank_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_page_buffer_bank_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.page_buffer_bank_busy_ns,
                r.hbf_stats.page_buffer_banks,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_page_buffer_bank_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.page_buffer_bank_busy_ns,
                r.hbf_stats.page_buffer_banks,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_flash_source_queue_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.flash_source_queue_busy_ns);
        })},
        {"hbf_flash_source_queue_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.flash_source_queue_resources);
        })},
        {"hbf_flash_source_queue_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_flash_source_queue_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.flash_source_queue_busy_ns,
                r.hbf_stats.flash_source_queue_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_flash_source_queue_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.flash_source_queue_busy_ns,
                r.hbf_stats.flash_source_queue_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_channel_command_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.channel_command_busy_ns);
        })},
        {"hbf_channel_command_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.channel_command_resources);
        })},
        {"hbf_channel_command_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_channel_command_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.channel_command_busy_ns,
                r.hbf_stats.channel_command_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_channel_command_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.channel_command_busy_ns,
                r.hbf_stats.channel_command_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_channel_data_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.channel_data_busy_ns);
        })},
        {"hbf_channel_data_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.channel_data_resources);
        })},
        {"hbf_channel_data_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_channel_data_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.channel_data_busy_ns,
                r.hbf_stats.channel_data_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_channel_data_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.channel_data_busy_ns,
                r.hbf_stats.channel_data_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_tsv_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.tsv_busy_ns);
        })},
        {"hbf_tsv_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.tsv_resources);
        })},
        {"hbf_tsv_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_tsv_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.tsv_busy_ns,
                r.hbf_stats.tsv_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_tsv_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.tsv_busy_ns,
                r.hbf_stats.tsv_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_sram_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.sram_busy_ns);
        })},
        {"hbf_sram_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.sram_resources);
        })},
        {"hbf_sram_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_sram_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.sram_busy_ns,
                r.hbf_stats.sram_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_sram_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.sram_busy_ns,
                r.hbf_stats.sram_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_hbio_command_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.hb_io_command_busy_ns);
        })},
        {"hbf_hbio_command_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.stacks);
        })},
        {"hbf_hbio_command_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_hbio_command_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.hb_io_command_busy_ns,
                r.hbf_stats.stacks,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_hbio_data_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.hb_io_data_busy_ns);
        })},
        {"hbf_hbio_data_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.stacks);
        })},
        {"hbf_hbio_data_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_hbio_data_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.hb_io_data_busy_ns,
                r.hbf_stats.stacks,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_sequencer_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.sequencer_busy_ns);
        })},
        {"hbf_sequencer_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.dies);
        })},
        {"hbf_sequencer_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_sequencer_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.sequencer_busy_ns,
                r.hbf_stats.dies,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_sequencer_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.sequencer_busy_ns,
                r.hbf_stats.dies,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_media_parallelism", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.media_parallelism(), 2)); })},
        {"hbf_active_planes", hbf_cell([](R r) {
            return cell_same(ratio_text(r.hbf_stats.active_planes, r.hbf_stats.planes)); })},
        {"hbf_read_lane_parallelism", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.read_lane_parallelism(), 2)); })},
        {"hbf_active_media_lanes", hbf_cell([](R r) {
            return cell_same(ratio_text(r.hbf_stats.active_media_lanes, r.hbf_stats.media_lanes)); })},
        {"hbf_media_lane_skew", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.media_lane_skew(), 2)); })},
        {"hbf_subarray_read_parallelism", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.subarray_read_parallelism(), 2)); })},
        {"hbf_active_subarrays", hbf_cell([](R r) {
            return cell_same(ratio_text(r.hbf_stats.active_subarrays, r.hbf_stats.subarrays)); })},
        {"hbf_subarray_busy_skew", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.subarray_busy_skew(), 2)); })},
        {"hbf_subarray_read_skew", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.subarray_read_skew(), 2)); })},
        {"hbf_page_buffer_bank_parallelism", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.page_buffer_bank_parallelism(), 2)); })},
        {"hbf_active_page_buffer_banks", hbf_cell([](R r) {
            return cell_same(ratio_text(r.hbf_stats.active_page_buffer_banks, r.hbf_stats.page_buffer_banks)); })},
        {"hbf_page_buffer_bank_skew", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.page_buffer_bank_skew(), 2)); })},
        {"hbf_page_buffer_bank_read_skew", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.page_buffer_bank_read_skew(), 2)); })},
        {"hbf_plane_media_skew", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.plane_media_skew(), 2)); })},
        {"hbf_io_util", hbf_cell([](R r) { return cell_percent(r.hbf_stats.io_utilization()); })},
        {"hbf_hbio_command_utilization", hbf_cell([](R r) {
            return cell_percent(r.hbf_stats.hbio_command_utilization()); })},
        {"hbf_hbio_data_utilization", hbf_cell([](R r) {
            return cell_percent(r.hbf_stats.hbio_data_utilization()); })},
        {"hbf_channel_parallelism", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.channel_parallelism(), 2)); })},
        {"hbf_active_channels", hbf_cell([](R r) {
            return cell_same(ratio_text(r.hbf_stats.active_channels, r.hbf_stats.channels)); })},
        {"hbf_sequencer_parallelism", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.sequencer_parallelism(), 2)); })},
        {"hbf_active_dies", hbf_cell([](R r) {
            return cell_same(ratio_text(r.hbf_stats.active_dies, r.hbf_stats.dies)); })},
        {"user_latency_average_ns", [](R r) {
            return cell_fixed(latency_distribution(r).average_ns);
        }},
        {"user_latency_p50_ns", [](R r) {
            return cell_fixed(latency_distribution(r).p50_ns);
        }},
        {"user_latency_p95_ns", [](R r) {
            return cell_fixed(latency_distribution(r).p95_ns);
        }},
        {"user_latency_max_ns", [](R r) {
            return cell_fixed(latency_distribution(r).max_ns);
        }},
        {"service_latency_average_ns", [](R r) {
            return cell_fixed(service_latency_distribution(r).average_ns);
        }},
        {"service_latency_p50_ns", [](R r) {
            return cell_fixed(service_latency_distribution(r).p50_ns);
        }},
        {"service_latency_p95_ns", [](R r) {
            return cell_fixed(service_latency_distribution(r).p95_ns);
        }},
        {"service_latency_max_ns", [](R r) {
            return cell_fixed(service_latency_distribution(r).max_ns);
        }},
        {"source_latency_average_ns", [](R r) {
            return cell_fixed(source_latency_distribution(r).average_ns);
        }},
        {"source_latency_p50_ns", [](R r) {
            return cell_fixed(source_latency_distribution(r).p50_ns);
        }},
        {"source_latency_p95_ns", [](R r) {
            return cell_fixed(source_latency_distribution(r).p95_ns);
        }},
        {"source_latency_max_ns", [](R r) {
            return cell_fixed(source_latency_distribution(r).max_ns);
        }},
        {"trace_origin_ns", [](R r) {
            return cell_fixed(r.first_arrival_ns);
        }},
        {"last_offered_arrival_ns", [](R r) {
            return cell_fixed(r.last_arrival_ns);
        }},
        {"last_user_completion_ns", [](R r) {
            return cell_fixed(r.user_finish_ns);
        }},
        {"quiescent_finish_ns", [](R r) {
            return cell_fixed(r.finish_ns);
        }},
        {"offered_arrival_span_ns", [](R r) {
            return cell_fixed(r.last_arrival_ns - r.first_arrival_ns);
        }},
        {"post_offer_user_completion_tail_ns", [](R r) {
            return cell_fixed(r.user_finish_ns - r.last_arrival_ns);
        }},
        {"user_completion_span_ns", [](R r) {
            return cell_fixed(r.user_finish_ns - r.first_arrival_ns);
        }},
        {"drain_tail_ns", [](R r) {
            return cell_fixed(r.finish_ns - r.user_finish_ns);
        }},
        {"makespan_ns", [](R r) {
            return cell_fixed(r.finish_ns - r.first_arrival_ns);
        }},
        {"service_to_user_completion_sum_work_ns", [](R r) {
            return cell_fixed(
                service_latency_distribution(r).sum_work_ns);
        }},
        {"phase_barriers", [](R r) {
            return cell_count(r.phase_barriers);
        }},
        {"phase_dependency_waited_ops", [](R r) {
            return cell_count(r.phase_dependency_waited_ops);
        }},
        {"phase_dependency_wait_work_ns", [](R r) {
            return cell_fixed(r.phase_dependency_wait_work_ns);
        }},
        {"phase_dependency_max_wait_ns", [](R r) {
            return cell_fixed(r.phase_dependency_max_wait_ns);
        }},
        {"front_end_admission_waited_ops", [](R r) {
            return cell_count(r.front_end_admission_waited_ops);
        }},
        {"front_end_admission_wait_work_ns", [](R r) {
            return cell_fixed(r.front_end_admission_wait_work_ns);
        }},
        {"front_end_admission_max_wait_ns", [](R r) {
            return cell_fixed(r.front_end_admission_max_wait_ns);
        }},
        {"offered_to_user_completion_sum_work_ns", [](R r) {
            return cell_fixed(latency_distribution(r).sum_work_ns);
        }},
        {"source_to_user_completion_sum_work_ns", [](R r) {
            return cell_fixed(
                source_latency_distribution(r).sum_work_ns);
        }},
        {"user_completion_throughput_GBps", [](R r) {
            return cell_fixed(user_completion_throughput_GBps(r)); }},
        {"makespan_throughput_GBps", [](R r) {
            return cell_fixed(makespan_throughput_GBps(r)); }},
        {"warnings", [](R r) { return cell_count(r.warnings.size()); }},
    };
    return columns;
}

void print_result_table(const std::vector<ScenarioResult>& results) {
    const auto& columns = result_columns();
    for (std::size_t i = 0; i < columns.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << columns[i].header;
    }
    std::cout << '\n';
    for (const auto& result : results) {
        for (std::size_t i = 0; i < columns.size(); ++i) {
            std::cout << (i == 0 ? "" : ",") << columns[i].cell(result).console;
        }
        std::cout << '\n';
    }
}

void print_time_breakdown(const ScenarioResult& result) {
    const auto time = make_time_breakdown(result);
    const auto& wall = time.wall_clock;
    const auto& latency = time.latency_work;
    std::cout << "  Time breakdown contract: wall spans are additive; "
                 "stage work may overlap and must not be summed as elapsed time.\n";
    std::cout << "    wall_clock_component,value_ns\n"
              << "    offered_arrival_span," << fixed(wall.offered_arrival_span_ns) << '\n'
              << "    post_offer_user_completion_tail,"
              << fixed(wall.post_offer_user_completion_tail_ns) << '\n'
              << "    user_completion_span," << fixed(wall.user_completion_span_ns) << '\n'
              << "    drain_tail," << fixed(wall.drain_tail_ns) << '\n'
              << "    makespan," << fixed(wall.makespan_ns) << '\n';
    std::cout << "    latency_work_component,value_ns\n"
              << "    service_to_user_completion_sum,"
              << fixed(latency.service_to_user_completion_sum_work_ns) << '\n'
              << "    phase_dependency_wait_sum,"
              << fixed(latency.phase_dependency_wait_work_ns) << '\n'
              << "    front_end_admission_wait_sum,"
              << fixed(latency.front_end_admission_wait_work_ns) << '\n'
              << "    offered_to_user_completion_sum,"
              << fixed(latency.offered_to_user_completion_sum_work_ns) << '\n'
              << "    source_to_user_completion_sum,"
              << fixed(latency.source_to_user_completion_sum_work_ns) << '\n'
              << "    phase_dependency_max,"
              << fixed(latency.phase_dependency_max_wait_ns) << '\n'
              << "    front_end_admission_max,"
              << fixed(latency.front_end_admission_max_wait_ns) << '\n';

    std::cout << "    stage_component,hbm_work_ns,hbf_work_ns,"
                 "external_work_ns\n";
    const auto stage_row = [&](const char* name, double Breakdown::*member) {
        std::cout << "    " << name << ','
                  << fixed(time.hbm_stage_work.*member) << ','
                  << fixed(time.hbf_stage_work.*member) << ','
                  << fixed(time.external_backing_stage_work.*member) << '\n';
    };
    stage_row("ingress_queue_wait", &Breakdown::ingress_queue_wait_ns);
    stage_row("scheduler_queue_wait", &Breakdown::scheduler_queue_wait_ns);
    stage_row("address_mapping", &Breakdown::address_mapping_ns);
    stage_row("translation", &Breakdown::translation_ns);
    stage_row("mapping_dram", &Breakdown::mapping_dram_ns);
    stage_row("refresh_stall", &Breakdown::refresh_stall_ns);
    stage_row("precharge", &Breakdown::precharge_ns);
    stage_row("activation", &Breakdown::activation_ns);
    stage_row("command", &Breakdown::command_ns);
    stage_row("array_read", &Breakdown::array_read_ns);
    stage_row("array_program", &Breakdown::array_program_ns);
    stage_row("array_erase", &Breakdown::array_erase_ns);
    stage_row("program_verify", &Breakdown::program_verify_ns);
    stage_row("media_lane_transfer", &Breakdown::media_lane_transfer_ns);
    stage_row("page_buffer", &Breakdown::page_buffer_ns);
    stage_row("sram_staging", &Breakdown::sram_staging_ns);
    stage_row("channel_transfer", &Breakdown::channel_transfer_ns);
    stage_row("tsv_transfer", &Breakdown::tsv_transfer_ns);
    stage_row("hb_io_transfer", &Breakdown::hb_io_transfer_ns);
    stage_row("transport_latency", &Breakdown::transport_latency_ns);
    stage_row("ecc_queue_wait", &Breakdown::ecc_queue_wait_ns);
    stage_row("ecc_response_latency", &Breakdown::ecc_latency_ns);
    stage_row("maintenance", &Breakdown::maintenance_ns);
    if (result.has_hbf) {
        const auto& hbf = result.hbf_stats;
        std::cout << "    hbf_subdiagnostic,decode,encode\n"
                  << "    ecc_queue_wait_work_ns,"
                  << fixed(hbf.ecc_decode_queue_wait_ns) << ','
                  << fixed(hbf.ecc_encode_queue_wait_ns) << '\n'
                  << "    ecc_response_latency_work_ns,"
                  << fixed(hbf.ecc_decode_latency_work_ns) << ','
                  << fixed(hbf.ecc_encode_latency_work_ns) << '\n'
                  << "    ecc_issue_busy_ns,"
                  << fixed(hbf.ecc_decode_issue_busy_ns) << ','
                  << fixed(hbf.ecc_encode_issue_busy_ns) << '\n'
                  << "    ecc_ops," << hbf.ecc_decode_ops << ','
                  << hbf.ecc_encode_ops << '\n'
                  << "    ecc_codeword_bytes,"
                  << hbf.ecc_decode_codeword_bytes << ','
                  << hbf.ecc_encode_codeword_bytes << '\n'
                  << "    hbf_write_buffer_slot_waited_ops,"
                  << hbf.write_buffer_slot_wait_ops << ",n/a\n"
                  << "    hbf_write_buffer_slot_wait_work_ns,"
                  << fixed(hbf.write_buffer_slot_wait_ns) << ",n/a\n";
    }

    const auto& link = time.base_die_link_stage_work;
    std::cout << "    base_die_link_component,work_ns\n"
              << "    read_queue_wait," << fixed(link.read_queue_wait_work_ns) << '\n'
              << "    write_queue_wait," << fixed(link.write_queue_wait_work_ns) << '\n'
              << "    read_serialization," << fixed(link.read_serialization_work_ns) << '\n'
              << "    write_serialization," << fixed(link.write_serialization_work_ns) << '\n'
              << "    read_fixed_latency," << fixed(link.read_fixed_latency_work_ns) << '\n'
              << "    write_fixed_latency," << fixed(link.write_fixed_latency_work_ns) << '\n';

    const auto& controller =
        time.layer_streaming_controller_stage_work;
    if (controller.present) {
        std::cout << "    layer_streaming_controller_component,work_ns\n"
                  << "    backing_admission_wait_work,"
                  << fixed(controller.backing_admission_wait_work_ns) << '\n'
                  << "    backing_admission_max_wait,"
                  << fixed(controller.backing_admission_max_wait_ns) << '\n'
                  << "    user_wait_work,"
                  << fixed(controller.user_wait_work_ns) << '\n'
                  << "    user_max_wait,"
                  << fixed(controller.user_max_wait_ns) << '\n'
                  << "    exposed_prefetch,"
                  << fixed(controller.exposed_prefetch_ns) << '\n'
                  << "    hidden_prefetch,"
                  << fixed(controller.hidden_prefetch_ns) << '\n'
                  << "    buffer_reuse_wait_work,"
                  << fixed(controller.buffer_reuse_wait_work_ns) << '\n';
    }
    const auto& cooperative = time.cooperative_write_controller_stage_work;
    if (cooperative.present) {
        std::cout << "    cooperative_write_controller_component,value\n"
                  << "    full_waited_ops," << cooperative.full_waited_ops << '\n'
                  << "    full_wait_work_ns,"
                  << fixed(cooperative.full_wait_work_ns) << '\n';
    }

    std::cout << "    resource,busy_ns,resource_count,active_span_ns,"
                 "capacity_time_ns,utilization\n";
    const auto resource_row = [](const char* name, const ResourceBusyMetric& metric) {
        std::cout << "    " << name << ',' << fixed(metric.busy_ns) << ','
                  << metric.resource_count << ',' << fixed(metric.active_span_ns)
                  << ',' << fixed(metric.capacity_time_ns) << ','
                  << fixed(metric.utilization, 6) << '\n';
    };
    const auto& resource = time.resource_busy;
    if (result.has_hbm) resource_row("hbm_data_bus", resource.hbm_data_bus);
    if (result.has_hbf) {
        resource_row("hbf_logic_ingress", resource.hbf_logic_ingress);
        resource_row(
            "hbf_mapping_dram_issue",
            resource.hbf_mapping_dram_issue);
        resource_row("hbf_plane_media", resource.hbf_plane_media);
        resource_row("hbf_media_lane", resource.hbf_media_lane);
        resource_row("hbf_subarray", resource.hbf_subarray);
        resource_row("hbf_page_buffer_bank", resource.hbf_page_buffer_bank);
        resource_row("hbf_flash_source_queue", resource.hbf_flash_source_queue);
        resource_row("hbf_channel_command", resource.hbf_channel_command);
        resource_row("hbf_channel_data", resource.hbf_channel_data);
        resource_row("hbf_tsv", resource.hbf_tsv);
        resource_row("hbf_sram", resource.hbf_sram);
        resource_row("hbf_hbio_command", resource.hbf_hbio_command);
        resource_row("hbf_hbio_data", resource.hbf_hbio_data);
        resource_row("hbf_sequencer", resource.hbf_sequencer);
        resource_row("hbf_ecc_issue", resource.hbf_ecc_issue);
    }
    if (result.has_external_backing) {
        resource_row(
            "external_controller", resource.external_controller);
        resource_row("external_media", resource.external_media);
        resource_row("external_link_m2s", resource.external_link_m2s);
        resource_row("external_link_s2m", resource.external_link_s2m);
    }
    if (result.base_die_link_stats.read_transfers != 0) {
        resource_row("base_die_link_read", resource.base_die_link_read);
    }
    if (result.base_die_link_stats.write_transfers != 0) {
        resource_row("base_die_link_write", resource.base_die_link_write);
    }
}

void print_address_heatmap_summary(const ScenarioResult& result) {
    if (!result.address_heatmap) {
        std::cout << "  Address heatmap: unavailable (this is a model error for "
                     "scenario_compare runs)\n";
        return;
    }
    std::cout << "  Address heatmap: logical traffic is request bytes; HBM physical "
                 "traffic is full bursts; HBF physical traffic is full pages/blocks.\n";
    std::cout << "    domain,size_bytes,read_bytes,write_bytes,erase_bytes,active_bins,"
                 "total_bins,hottest_read_range,hottest_read_bytes,"
                 "hottest_write_range,hottest_write_bytes\n";
    for (const auto& domain : result.address_heatmap->domains) {
        std::size_t active_bins = 0;
        const hbfsim::physical::AddressHeatmapBin* hottest_read = nullptr;
        const hbfsim::physical::AddressHeatmapBin* hottest_write = nullptr;
        for (const auto& bin : domain.bins) {
            if (bin.total.read_bytes != 0 || bin.total.write_bytes != 0 ||
                bin.total.erase_bytes != 0) {
                ++active_bins;
            }
            if (bin.total.read_bytes != 0 && (hottest_read == nullptr ||
                bin.total.read_bytes > hottest_read->total.read_bytes)) {
                hottest_read = &bin;
            }
            if (bin.total.write_bytes != 0 && (hottest_write == nullptr ||
                bin.total.write_bytes > hottest_write->total.write_bytes)) {
                hottest_write = &bin;
            }
        }
        const auto range = [](const hbfsim::physical::AddressHeatmapBin* bin) {
            if (bin == nullptr) {
                return std::string{"n/a"};
            }
            std::ostringstream text;
            text << "[0x" << std::hex << bin->begin << ",0x";
            if (bin->end.is_full_address_space_end()) {
                text << "10000000000000000";
            } else {
                text << bin->end.finite_value();
            }
            text << ')';
            return text.str();
        };
        std::cout << "    " << hbfsim::physical::to_string(domain.domain) << ','
                  << hbfsim::physical::address_boundary_decimal(domain.size_bytes) << ','
                  << domain.total.read_bytes << ','
                  << domain.total.write_bytes << ',' << domain.total.erase_bytes << ','
                  << active_bins << ',' << domain.bins.size() << ','
                  << range(hottest_read) << ','
                  << (hottest_read == nullptr ? 0 : hottest_read->total.read_bytes) << ','
                  << range(hottest_write) << ','
                  << (hottest_write == nullptr ? 0 : hottest_write->total.write_bytes)
                  << '\n';
    }
}

void print_device_stats(const ScenarioResult& result) {
    std::cout << "\n[" << result.name << "]\n";
    if (result.has_hbm) {
        const auto& s = result.hbm_stats;
        std::cout << "  HBM: reads=" << s.read_bytes
                  << " writes=" << s.write_bytes
                  << " row_hits=" << s.row_hits
                  << " row_misses=" << s.row_misses
                  << " row_conflicts=" << s.row_conflicts
                  << " hit_rate=" << fixed(s.row_hit_rate() * 100.0) << "%"
                  << " scheduler_queue_wait_work_ns="
                  << fixed(s.stage_work.scheduler_queue_wait_ns)
                  << " max_queue_occupancy=" << s.max_queue_occupancy
                  << " bus_busy_ns=" << fixed(s.bus_busy_ns)
                  << " util=" << fixed(s.utilization() * 100.0) << "%"
                  << " bus_parallelism=" << fixed(s.bus_parallelism(), 2)
                  << " active_pch=" << s.active_pseudo_channels << "/" << s.pseudo_channels
                  << " pch_busy_skew=" << fixed(s.pseudo_channel_busy_skew(), 2)
                  << " finish_ns=" << fixed(s.finish_ns)
                  << '\n';
    }
    if (result.has_hbf) {
        const auto& s = result.hbf_stats;
        std::cout << "  HBF: logical_read=" << s.logical_read_bytes
                  << " physical_read=" << s.physical_read_bytes
                  << " logical_write=" << s.logical_write_bytes
                  << " physical_write=" << s.physical_write_bytes
                  << " page_reads=" << s.page_reads
                  << " data_programs=" << s.data_programs
                  << " page_programs=" << s.page_programs
                  << " block_erases=" << s.block_erases
                  << " mappings=" << s.mapping_entries
                  << " total_pages=" << s.total_pages
                  << " free_pages=" << s.free_pages
                  << " valid_pages=" << s.valid_pages
                  << " invalid_pages=" << s.invalid_pages
                  << " pending_program_pages=" << s.pending_program_pages
                  << " pending_mapping_publications="
                  << s.pending_mapping_publications
                  << " static_unmaterialized_pages="
                  << s.static_unmaterialized_pages
                  << " accounting_verified="
                  << (s.accounting_verified ? "yes" : "no")
                  << " waf="
                  << (s.waf() ? fixed(*s.waf(), 3) : "n/a")
                  << " resident_mapping_bytes="
                  << s.resident_mapping_table_bytes
                  << " resident_mapping_bytes_per_stack="
                  << s.resident_mapping_table_bytes_per_stack
                  << " resident_mapping_pages_per_stack="
                  << s.resident_mapping_pages_per_stack
                  << " mapping_lookups=" << s.mapping_lookup_ops
                  << " mapping_user_lookups=" << s.mapping_user_lookup_ops
                  << " mapping_gc_lookups=" << s.mapping_gc_lookup_ops
                  << " mapping_updates=" << s.mapping_update_ops
                  << " mapping_user_updates=" << s.mapping_user_update_ops
                  << " mapping_gc_updates=" << s.mapping_gc_update_ops
                  << " mapping_dram_waits=" << s.mapping_dram_wait_ops
                  << " mapping_dram_wait_work_ns="
                  << fixed(s.mapping_dram_wait_ns)
                  << " mapping_dram_wait_max_ns="
                  << fixed(s.mapping_dram_wait_max_ns)
                  << " mapping_dram_issue_busy_ns="
                  << fixed(s.mapping_dram_issue_busy_ns)
                  << " page_read_admission_waited="
                  << s.page_read_admission_waited_pages
                  << " page_read_admission_wait_ns="
                  << fixed(s.page_read_admission_wait_ns)
                  << " page_read_admission_max_wait_ns="
                  << fixed(s.page_read_admission_max_wait_ns)
                  << " mapping_programs=" << s.mapping_page_programs
                  << " read_buffer_hits=" << s.read_buffer_hits
                  << " write_buffer_hits=" << s.write_buffer_hits
                  << " write_buffer_misses=" << s.write_buffer_misses
                  << " write_buffer_flushes=" << s.write_buffer_flushes
                  << " write_buffer_read_hits=" << s.write_buffer_read_hits
                  << " write_buffer_read_bytes=" << s.write_buffer_read_bytes
                  << " write_buffer_slot_waits=" << s.write_buffer_slot_wait_ops
                  << " write_buffer_slot_wait_work_ns="
                  << fixed(s.write_buffer_slot_wait_ns)
                  << " read_splits=" << s.read_splits
                  << " flash_sched_enq=" << s.flash_scheduler_enqueues
                  << " flash_sched_issue=" << s.flash_scheduler_issues
                  << " gc_runs=" << s.gc_runs
                  << " gc_relocations=" << s.gc_relocations
                  << " gc_data_relocations=" << s.gc_data_relocations
                  << " gc_mapping_relocations=" << s.gc_mapping_relocations
                  << " gc_reclaimed_invalid_pages="
                  << s.gc_reclaimed_invalid_pages
                  << " ingress_queue_wait_work_ns="
                  << fixed(s.stage_work.ingress_queue_wait_ns)
                  << " scheduler_queue_wait_work_ns="
                  << fixed(s.stage_work.scheduler_queue_wait_ns)
                  << " ecc_queue_wait_work_ns="
                  << fixed(s.stage_work.ecc_queue_wait_ns)
                  << " ecc_decode_ops=" << s.ecc_decode_ops
                  << " ecc_encode_ops=" << s.ecc_encode_ops
                  << " ecc_codeword_bytes=" << s.ecc_codeword_bytes
                  << " ecc_issue_busy_ns=" << fixed(s.ecc_issue_busy_ns)
                  << " ecc_issue_util=" << fixed(s.ecc_issue_utilization() * 100.0) << "%"
                  << " ecc_issue_parallelism=" << fixed(s.ecc_issue_parallelism(), 2)
                  << " ecc_active_dies=" << s.active_ecc_dies << "/" << s.dies
                  << " ecc_max_inflight_per_die=" << s.max_ecc_inflight_per_die
                  << " array_read_work_ns=" << fixed(s.stage_work.array_read_ns)
                  << " array_program_work_ns="
                  << fixed(s.stage_work.array_program_ns)
                  << " program_verify_work_ns="
                  << fixed(s.stage_work.program_verify_ns)
                  << " array_erase_work_ns=" << fixed(s.stage_work.array_erase_ns)
                  << " media_util=" << fixed(s.media_utilization() * 100.0) << "%"
                  << " media_parallelism=" << fixed(s.media_parallelism(), 2)
                  << " read_lane_parallelism=" << fixed(s.read_lane_parallelism(), 2)
                  << " subarray_read_parallelism=" << fixed(s.subarray_read_parallelism(), 2)
                  << " page_buffer_bank_parallelism="
                  << fixed(s.page_buffer_bank_parallelism(), 2)
                  << " active_planes=" << s.active_planes << "/" << s.planes
                  << " active_media_lanes=" << s.active_media_lanes << "/" << s.media_lanes
                  << " active_subarrays=" << s.active_subarrays << "/" << s.subarrays
                  << " active_page_buffer_banks=" << s.active_page_buffer_banks
                  << "/" << s.page_buffer_banks
                  << " plane_media_skew=" << fixed(s.plane_media_skew(), 2)
                  << " media_lane_skew=" << fixed(s.media_lane_skew(), 2)
                  << " subarray_busy_skew=" << fixed(s.subarray_busy_skew(), 2)
                  << " subarray_read_skew=" << fixed(s.subarray_read_skew(), 2)
                  << " page_buffer_bank_skew=" << fixed(s.page_buffer_bank_skew(), 2)
                  << " page_buffer_bank_read_skew="
                  << fixed(s.page_buffer_bank_read_skew(), 2)
                  << " io_util=" << fixed(s.io_utilization() * 100.0) << "%"
                  << " hbio_cmd_util="
                  << fixed(s.hbio_command_utilization() * 100.0) << "%"
                  << " hbio_data_util="
                  << fixed(s.hbio_data_utilization() * 100.0) << "%"
                  << " channel_parallelism=" << fixed(s.channel_parallelism(), 2)
                  << " active_channels=" << s.active_channels << "/" << s.channels
                  << " sequencer_parallelism=" << fixed(s.sequencer_parallelism(), 2)
                  << " active_dies=" << s.active_dies << "/" << s.dies
                  << " finish_ns=" << fixed(s.finish_ns)
                  << '\n';
    }
    if (result.has_external_backing) {
        const auto& s = result.external_backing_stats;
        std::cout << "  External backing: kind="
                  << hbfsim::physical::external::to_string(s.kind)
                  << " reads=" << s.read_requests
                  << " writes=" << s.write_requests
                  << " read_bytes=" << s.read_bytes
                  << " write_bytes=" << s.write_bytes
                  << " active_media_channels="
                  << s.active_media_channels
                  << "/" << s.media_channels
                  << " max_device_outstanding="
                  << s.max_device_outstanding
                  << " outstanding_wait_work_ns="
                  << fixed(s.outstanding_wait_ns)
                  << " controller_queue_wait_work_ns="
                  << fixed(s.controller_queue_wait_ns)
                  << " controller_util="
                  << fixed(s.controller_utilization() * 100.0) << "%"
                  << " media_queue_wait_work_ns="
                  << fixed(s.media_queue_wait_ns)
                  << " media_read_latency_work_ns="
                  << fixed(s.media_read_latency_work_ns)
                  << " media_write_latency_work_ns="
                  << fixed(s.media_write_latency_work_ns)
                  << " media_read_busy_ns="
                  << fixed(s.media_read_busy_ns)
                  << " media_write_busy_ns="
                  << fixed(s.media_write_busy_ns)
                  << " media_util="
                  << fixed(s.media_utilization() * 100.0) << "%"
                  << " m2s_wire_bytes=" << s.m2s_wire_bytes
                  << " s2m_wire_bytes=" << s.s2m_wire_bytes
                  << " m2s_util="
                  << fixed(s.m2s_utilization() * 100.0) << "%"
                  << " s2m_util="
                  << fixed(s.s2m_utilization() * 100.0) << "%"
                  << " finish_ns=" << fixed(s.finish_ns)
                  << '\n';
    }
    if (result.hbm_write_buffer_user_write_bytes != 0 ||
        result.hbm_write_buffer_peak_bytes != 0) {
        std::cout << "  Cooperative write region: user_write_bytes="
                  << result.hbm_write_buffer_user_write_bytes
                  << " destaged_bytes=" << result.hbm_write_buffer_destaged_bytes
                  << " peak_bytes=" << result.hbm_write_buffer_peak_bytes
                  << " full_waits=" << result.hbm_write_buffer_full_waits
                  << " wait_work_ns=" << fixed(result.hbm_write_buffer_wait_ns);
        if (result.hbm_write_buffer_user_write_bytes != 0 && result.has_hbf) {
            std::cout << " end_to_end_media_bytes_per_user_write_byte="
                      << fixed(
                          static_cast<double>(result.hbf_stats.physical_write_bytes) /
                          static_cast<double>(
                              result.hbm_write_buffer_user_write_bytes),
                          3);
        }
        std::cout << '\n';
    }
    for (const auto& warning : result.warnings) {
        std::cout << "  warning: " << warning << '\n';
    }
    if (result.behavioral_tiering_stats) {
        const auto& s = *result.behavioral_tiering_stats;
        std::cout << "  Behavioral tiering: policy="
                  << hbfsim::physical::hybrid::to_string(
                         s.admission_policy)
                  << " semantic_inputs_consumed="
                  << (s.semantic_inputs_consumed ? "yes" : "no")
                  << " hbm_tier_pages=" << s.hbm_tier_pages
                  << " history_capacity_pages="
                  << s.history_capacity_pages
                  << " observations=" << s.page_observations
                  << " hbm_hits=" << s.hbm_hits
                  << " hbf_bypasses=" << s.hbf_bypasses
                  << " promotions=" << s.promotions
                  << " clean_evictions=" << s.clean_evictions
                  << " dirty_evictions=" << s.dirty_evictions
                  << " backing_fill_bytes=" << s.backing_fill_bytes
                  << " dirty_writeback_bytes="
                  << s.dirty_writeback_bytes
                  << " transition_waited_transactions="
                  << s.transition_waited_transactions
                  << " transition_wait_work_ns="
                  << fixed(s.transition_wait_work_ns)
                  << " final_resident_pages="
                  << s.final_resident_pages
                  << " final_dirty_pages="
                  << s.final_dirty_pages
                  << " decision_fingerprint="
                  << s.decision_fingerprint
                  << '\n';
    }
    if (result.layer_streaming_stats) {
        const auto& s = *result.layer_streaming_stats;
        std::cout << "  Layer streaming: mode="
                  << layer_streaming_mode_name(s)
                  << " backing="
                  << hbfsim::physical::hybrid::to_string(s.backing)
                  << " residency_policy="
                  << layer_streaming_residency_policy_name(s)
                  << " compact_resident_mapping="
                  << (s.compact_resident_mapping ? "yes" : "no")
                  << " semantic_inputs_consumed="
                  << (s.semantic_inputs_consumed ? "yes" : "no")
                  << " explicit_residency_contract="
                  << (s.explicit_residency_contract ? "yes" : "no")
                  << " address_footprint_bytes="
                  << s.address_footprint_bytes
                  << " unique_resident_footprint_pages="
                  << s.unique_resident_footprint_pages
                  << " unique_resident_footprint_bytes="
                  << s.unique_resident_footprint_bytes
                  << " capacity_pressure_basis_bytes="
                  << s.capacity_pressure_basis_bytes
                  << " footprint_page_rounding_bytes="
                  << s.footprint_page_rounding_bytes
                  << " hbm_capacity_bytes=" << s.hbm_capacity_bytes
                  << " hbm_capacity_pressure="
                  << fixed(s.hbm_capacity_pressure, 6)
                  << " layers=" << s.layers
                  << " explicit_layer_requests=" << s.explicit_layer_requests
                  << " explicit_compute_layers="
                  << s.explicit_compute_layers
                  << " compute_work_ns=" << fixed(s.compute_work_ns)
                  << " hbm_only_resident_pages="
                  << s.hbm_only_resident_pages
                  << " hbm_only_resident_bytes="
                  << s.hbm_only_resident_bytes
                  << " hot_kv_candidate_pages="
                  << s.hot_kv_candidate_pages
                  << " hot_kv_candidate_bytes="
                  << s.hot_kv_candidate_bytes
                  << " hot_kv_resident_pages="
                  << s.hot_kv_resident_pages
                  << " hot_kv_resident_bytes="
                  << s.hot_kv_resident_bytes
                  << " data_pages=" << s.data_pages
                  << " data_bytes=" << s.data_bytes
                  << " model_weight_resident_pages="
                  << s.model_weight_resident_pages
                  << " model_weight_resident_bytes="
                  << s.model_weight_resident_bytes
                  << " model_weight_backing_pages="
                  << s.model_weight_backing_pages
                  << " model_weight_backing_bytes="
                  << s.model_weight_backing_bytes
                  << " cold_kv_backing_pages="
                  << s.cold_kv_backing_pages
                  << " cold_kv_backing_bytes="
                  << s.cold_kv_backing_bytes
                  << " unknown_backing_pages="
                  << s.unknown_backing_pages
                  << " unknown_backing_bytes="
                  << s.unknown_backing_bytes
                  << " backing_unique_pages="
                  << s.backing_unique_pages
                  << " backing_unique_bytes="
                  << s.backing_unique_bytes
                  << " resident_physical_pages="
                  << s.resident_physical_pages
                  << " resident_physical_bytes="
                  << s.resident_physical_bytes
                  << " effective_layer_buffer_pages="
                  << s.effective_layer_buffer_pages
                  << " effective_layer_buffer_bytes="
                  << s.effective_layer_buffer_bytes
                  << " unused_hbm_pages="
                  << s.unused_hbm_pages
                  << " unused_hbm_bytes="
                  << s.unused_hbm_bytes
                  << " streamed_pages=" << s.streamed_pages
                  << " streamed_bytes=" << s.streamed_bytes
                  << " foreground_resident_page_accesses="
                  << s.foreground_resident_page_accesses
                  << " foreground_buffer_page_accesses="
                  << s.foreground_buffer_page_accesses
                  << " dirty_pages_written_back="
                  << s.dirty_pages_written_back
                  << " writeback_bytes=" << s.writeback_bytes
                  << " max_layer_data_pages="
                  << s.max_layer_data_pages
                  << " max_layer_data_bytes="
                  << s.max_layer_data_bytes
                  << " immutable_weight_logical_bytes="
                  << s.immutable_weight_logical_bytes
                  << " runtime_overhead_logical_bytes="
                  << s.runtime_overhead_logical_bytes
                  << " block_table_logical_bytes="
                  << s.block_table_logical_bytes
                  << " active_buffer_logical_bytes_per_slot="
                  << s.active_buffer_logical_bytes_per_slot
                  << " residency_page_size_bytes="
                  << s.residency_page_size_bytes
                  << " kv_block_stride_bytes="
                  << s.kv_block_stride_bytes
                  << " logical_kv_blocks="
                  << s.logical_kv_blocks
                  << " hot_kv_blocks="
                  << s.hot_kv_blocks
                  << " cold_kv_blocks="
                  << s.cold_kv_blocks
                  << " backing_request_credit_limit="
                  << s.backing_request_credit_limit
                  << " backing_max_inflight_requests="
                  << s.backing_max_inflight_requests
                  << " backing_admission_waited_requests="
                  << s.backing_admission_waited_requests
                  << " backing_admission_wait_work_ns="
                  << fixed(s.backing_admission_wait_work_ns)
                  << " backing_admission_max_wait_ns="
                  << fixed(s.backing_admission_max_wait_ns)
                  << " user_waited_ops=" << s.user_waited_ops
                  << " user_wait_work_ns=" << fixed(s.user_wait_work_ns)
                  << " user_max_wait_ns=" << fixed(s.user_max_wait_ns)
                  << " exposed_prefetch_ns=" << fixed(s.exposed_prefetch_ns)
                  << " hidden_prefetch_ns=" << fixed(s.hidden_prefetch_ns)
                  << " buffer_reuse_wait_work_ns="
                  << fixed(s.buffer_reuse_wait_work_ns)
                  << " background_hbf_writes=" << result.background_hbf_writes
                  << " hbm_user_accesses=" << result.hbm_user_accesses
                  << " hbm_background_accesses=" << result.hbm_background_accesses
                  << " hbf_user_accesses=" << result.hbf_user_accesses
                  << " hbf_background_accesses=" << result.hbf_background_accesses
                  << " hbf_static_read_bytes=" << result.hbf_static_read_bytes
                  << " hbf_logical_read_bytes=" << result.hbf_logical_read_bytes
                  << " hbf_backing_write_bytes=" << result.hbf_backing_write_bytes
                  << " external_background_accesses="
                  << result.external_background_accesses
                  << " external_backing_read_bytes="
                  << result.external_backing_read_bytes
                  << " external_backing_write_bytes="
                  << result.external_backing_write_bytes
                  << " hbm_streaming_write_bytes="
                  << result.hbm_streaming_write_bytes
                  << " base_link_read_bytes="
                  << result.base_die_link_stats.read_bytes
                  << " base_link_write_bytes="
                  << result.base_die_link_stats.write_bytes
                  << " base_link_read_util="
                  << fixed(result.base_die_link_stats.read_utilization() * 100.0)
                  << "%"
                  << " base_link_write_util="
                  << fixed(result.base_die_link_stats.write_utilization() * 100.0)
                  << "%"
                  << '\n';
    }
    print_time_breakdown(result);
    print_address_heatmap_summary(result);
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string out = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            out += "\"\"";
        } else {
            out += ch;
        }
    }
    out += '"';
    return out;
}

void write_resolved_config(const std::string& output_path, const Options& options) {
    ensure_parent_dir(output_path);
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot open config output: " + output_path);
    }
    // A resolved config is a replay artifact, not presentation text. Preserve
    // every binary64 value exactly enough for parse -> write -> parse round
    // trips, including non-integer ECC raw-bandwidth assumptions.
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    const auto external = make_external_backing_config(options);
    out << "# HBFSim scenario_compare " << HBFSIM_VERSION << " resolved config\n";
    if (options.trace_path) out << "trace=" << *options.trace_path << '\n';
    if (options.expected_trace_sha256) {
        out << "expected-trace-sha256="
            << *options.expected_trace_sha256 << '\n';
    }
    if (options.expected_trace_file_bytes) {
        out << "expected-trace-bytes="
            << *options.expected_trace_file_bytes << '\n';
    }
    if (options.initial_image_trace_path) {
        out << "initial-image-trace="
            << *options.initial_image_trace_path << '\n';
    }
    if (options.synthetic_sequential_read_bytes) {
        out << "synthetic-sequential-read-bytes="
            << *options.synthetic_sequential_read_bytes << '\n';
        out << "synthetic-sequential-read-base="
            << options.synthetic_sequential_read_base << '\n';
    }
    if (options.generate_llm_path) out << "generate-llm=" << *options.generate_llm_path << '\n';
    if (options.generate_semantic_llm_path) {
        out << "generate-semantic-llm=" << *options.generate_semantic_llm_path << '\n';
    }
    if (options.chrome_trace_path) out << "chrome-trace=" << *options.chrome_trace_path << '\n';
    if (options.summary_csv_path) out << "summary-csv=" << *options.summary_csv_path << '\n';
    if (options.summary_json_path) out << "summary-json=" << *options.summary_json_path << '\n';
    out << "trace-census-only="
        << (options.trace_census_only ? "true" : "false") << '\n';
    if (!options.scenarios.empty()) {
        out << "scenarios=";
        for (std::size_t i = 0; i < options.scenarios.size(); ++i) {
            if (i != 0) out << ',';
            out << options.scenarios[i];
        }
        out << '\n';
    }
    out << "line-size=" << options.line_size << '\n';
    out << "interarrival-ns=" << options.interarrival_ns << '\n';
    out << "max-ops=" << options.max_ops << '\n';
    out << "hbm-capacity-bytes=" << options.hbm_capacity_bytes << '\n';
    out << "flat-hbm-bytes=" << options.flat_hbm_bytes << '\n';
    if (options.hbf_capacity_bytes_target) {
        out << "hbf-capacity-bytes=" << *options.hbf_capacity_bytes_target << '\n';
    }
    out << "layer-buffer-bytes=" << options.layer_buffer_bytes << '\n';
    out << "explicit-residency-contract="
        << (options.explicit_residency_contract ? "true" : "false") << '\n';
    if (options.explicit_residency_contract) {
        out << "residency-page-size-bytes="
            << *options.residency_page_size_bytes << '\n';
        out << "residency-unique-footprint-bytes="
            << *options.residency_unique_footprint_bytes << '\n';
        out << "residency-immutable-weight-bytes="
            << *options.residency_immutable_weight_bytes << '\n';
        out << "residency-immutable-weight-pages="
            << *options.residency_immutable_weight_pages << '\n';
        out << "residency-static-weight-pages="
            << *options.residency_static_weight_pages << '\n';
        out << "residency-runtime-overhead-bytes="
            << *options.residency_runtime_overhead_bytes << '\n';
        out << "residency-block-table-bytes="
            << *options.residency_block_table_bytes << '\n';
        out << "residency-active-buffer-bytes="
            << *options.residency_active_buffer_bytes << '\n';
        out << "residency-kv-region-begin="
            << *options.residency_kv_region_begin << '\n';
        out << "residency-kv-block-stride-bytes="
            << *options.residency_kv_block_stride_bytes << '\n';
        out << "residency-logical-kv-blocks="
            << *options.residency_logical_kv_blocks << '\n';
        out << "residency-hot-kv-blocks="
            << *options.residency_hot_kv_blocks << '\n';
    }
    out << "behavioral-hbm-bytes="
        << options.behavioral_hbm_bytes << '\n';
    out << "behavioral-promotion-threshold="
        << options.behavioral_promotion_threshold << '\n';
    out << "behavioral-history-pages="
        << options.behavioral_history_pages << '\n';
    out << "max-outstanding-requests=" << options.max_outstanding_requests << '\n';
    out << "max-hbm-outstanding-requests="
        << options.max_hbm_outstanding_requests << '\n';
    out << "max-hbf-outstanding-requests="
        << options.max_hbf_outstanding_requests << '\n';
    out << "address-heatmap-bins=" << options.address_heatmap_bins << '\n';
    out << "base-die-link-read-bw=" << options.base_die_link_read_bw_GBps << '\n';
    out << "base-die-link-write-bw=" << options.base_die_link_write_bw_GBps << '\n';
    out << "base-die-link-latency-ns=" << options.base_die_link_latency_ns << '\n';
    out << "external-backing-kind="
        << hbfsim::physical::external::to_string(external.kind) << '\n';
    out << "external-backing-capacity-bytes="
        << external.capacity_bytes << '\n';
    out << "external-backing-page-size="
        << external.page_size_bytes << '\n';
    out << "external-backing-media-channels="
        << external.media_channels << '\n';
    out << "external-backing-max-outstanding-requests="
        << external.max_outstanding_requests << '\n';
    out << "external-backing-controller-issue-ns="
        << external.controller_issue_ns << '\n';
    out << "external-backing-controller-processing-ns="
        << external.controller_processing_ns << '\n';
    out << "external-backing-media-read-latency-ns="
        << external.media_read_latency_ns << '\n';
    out << "external-backing-media-write-latency-ns="
        << external.media_write_latency_ns << '\n';
    out << "external-backing-media-read-bw="
        << external.media_read_bandwidth_GBps << '\n';
    out << "external-backing-media-write-bw="
        << external.media_write_bandwidth_GBps << '\n';
    out << "external-backing-m2s-bw="
        << external.m2s_bandwidth_GBps << '\n';
    out << "external-backing-s2m-bw="
        << external.s2m_bandwidth_GBps << '\n';
    out << "external-backing-one-way-propagation-ns="
        << external.one_way_propagation_ns << '\n';
    out << "external-backing-command-bytes="
        << external.command_bytes << '\n';
    out << "external-backing-completion-bytes="
        << external.completion_bytes << '\n';
    if (options.generate_llm_path || options.generate_semantic_llm_path) {
        out << "llm-tokens=" << options.llm_tokens << '\n';
        out << "llm-layers=" << options.llm_layers << '\n';
        out << "llm-weight-base=" << options.llm_weight_base << '\n';
        out << "llm-kv-base=" << options.llm_kv_base << '\n';
        out << "llm-scratch-base=" << options.llm_scratch_base << '\n';
    }
    out << "hbm-stacks=" << options.hbm_stacks << '\n';
    out << "hbm-channels=" << options.hbm_channels << '\n';
    out << "hbm-pseudo-channels=" << options.hbm_pseudo_channels << '\n';
    out << "hbm-bank-groups-per-pseudo-channel="
        << options.hbm_bank_groups_per_pseudo_channel << '\n';
    out << "hbm-banks-per-group=" << options.hbm_banks_per_group << '\n';
    out << "hbm-channel-row-size-bytes="
        << options.hbm_channel_row_size_bytes << '\n';
    out << "hbm-channel-width-bits=" << options.hbm_channel_width_bits << '\n';
    out << "hbm-burst-length=" << options.hbm_burst_length << '\n';
    out << "hbm-pin-rate-gbps=" << options.hbm_pin_rate_Gbps << '\n';
    out << "hbm-data-rate-per-command-clock="
        << options.hbm_data_rate_per_command_clock << '\n';
    out << "hbm-address-mapping-ns=" << options.hbm_address_mapping_ns << '\n';
    out << "hbm-trcdrd-ns=" << options.hbm_trcdrd_ns << '\n';
    out << "hbm-trcdwr-ns=" << options.hbm_trcdwr_ns << '\n';
    out << "hbm-tcl-ns=" << options.hbm_tcl_ns << '\n';
    out << "hbm-tcwl-ns=" << options.hbm_tcwl_ns << '\n';
    out << "hbm-trp-ns=" << options.hbm_trp_ns << '\n';
    out << "hbm-tras-ns=" << options.hbm_tras_ns << '\n';
    out << "hbm-trc-ns=" << options.hbm_trc_ns << '\n';
    out << "hbm-twr-ns=" << options.hbm_twr_ns << '\n';
    out << "hbm-trtp-ns=" << options.hbm_trtp_ns << '\n';
    out << "hbm-tccd-s-cycles=" << options.hbm_tccd_s_cycles << '\n';
    out << "hbm-tccd-l-cycles=" << options.hbm_tccd_l_cycles << '\n';
    out << "hbm-trrd-s-ns=" << options.hbm_trrd_s_ns << '\n';
    out << "hbm-trrd-l-ns=" << options.hbm_trrd_l_ns << '\n';
    out << "hbm-tfaw-ns=" << options.hbm_tfaw_ns << '\n';
    out << "hbm-twtr-s-ns=" << options.hbm_twtr_s_ns << '\n';
    out << "hbm-twtr-l-ns=" << options.hbm_twtr_l_ns << '\n';
    out << "hbm-trtw-ns=" << options.hbm_trtw_ns << '\n';
    out << "hbm-refresh=" << (options.hbm_refresh_enabled ? "true" : "false") << '\n';
    out << "hbm-same-bank-refresh=" << (options.hbm_same_bank_refresh ? "true" : "false") << '\n';
    out << "hbm-trefi-ns=" << options.hbm_trefi_ns << '\n';
    out << "hbm-trfc-ns=" << options.hbm_trfc_ns << '\n';
    out << "hbm-trfcsb-ns=" << options.hbm_trfcsb_ns << '\n';
    out << "hbm-queue-depth=" << options.hbm_queue_depth << '\n';
    out << "hbm-frfcfs-cap-ns=" << options.hbm_frfcfs_cap_ns << '\n';
    out << "hbf-stacks=" << options.hbf_stacks << '\n';
    out << "hbf-channels=" << options.hbf_channels << '\n';
    out << "hbf-dies-per-channel=" << options.hbf_dies_per_channel << '\n';
    out << "hbf-planes-per-die=" << options.hbf_planes_per_die << '\n';
    out << "hbf-blocks-per-plane=" << options.hbf_blocks_per_plane << '\n';
    out << "hbf-pages-per-block=" << options.hbf_pages_per_block << '\n';
    out << "hbf-page-size=" << options.hbf_page_size << '\n';
    out << "hbf-oob-bytes=" << options.hbf_oob_bytes << '\n';
    out << "hbf-media-lanes-per-plane=" << options.hbf_media_lanes_per_plane << '\n';
    out << "hbf-subarrays-per-plane=" << options.hbf_subarrays_per_plane << '\n';
    out << "hbf-page-buffer-banks-per-plane="
        << options.hbf_page_buffer_banks_per_plane << '\n';
    out << "hbf-read-ns=" << options.hbf_read_ns << '\n';
    out << "hbf-program-ns=" << options.hbf_program_ns << '\n';
    out << "hbf-program-verify-ns=" << options.hbf_program_verify_ns << '\n';
    out << "hbf-erase-ns=" << options.hbf_erase_ns << '\n';
    out << "hbf-ecc-decode-latency-ns="
        << options.hbf_ecc_decode_latency_ns << '\n';
    out << "hbf-ecc-encode-latency-ns="
        << options.hbf_ecc_encode_latency_ns << '\n';
    out << "hbf-ecc-decode-raw-bw="
        << options.hbf_ecc_decode_raw_bw_GBps_per_die << '\n';
    out << "hbf-ecc-encode-raw-bw="
        << options.hbf_ecc_encode_raw_bw_GBps_per_die << '\n';
    out << "hbf-channel-bw=" << options.hbf_channel_bw_GBps << '\n';
    out << "hbf-hbio-bw=" << options.hbf_hbio_bw_GBps << '\n';
    out << "hbf-tsv-bw=" << options.hbf_tsv_bw_GBps << '\n';
    out << "hbf-media-lane-bw=" << options.hbf_media_lane_bw_GBps << '\n';
    out << "hbf-logic-sram-bw=" << options.hbf_logic_sram_bw_GBps << '\n';
    out << "hbf-page-buffer-bw=" << options.hbf_page_buffer_bw_GBps << '\n';
    out << "hbf-ctrl-dram-bytes=" << options.hbf_ctrl_dram_bytes << '\n';
    out << "hbf-ctrl-dram-latency-ns=" << options.hbf_ctrl_dram_latency_ns << '\n';
    out << "hbf-ctrl-dram-issue-ns=" << options.hbf_ctrl_dram_issue_ns << '\n';
    out << "hbf-flash-tsu-issue-ns=" << options.hbf_flash_tsu_issue_ns << '\n';
    out << "hbf-logic-scheduler-issue-ns=" << options.hbf_logic_scheduler_issue_ns << '\n';
    out << "hbf-batch-activation=" << (options.hbf_batch_activation ? "true" : "false") << '\n';
    out << "hbf-read-buffer-pages=" << options.hbf_read_buffer_pages << '\n';
    out << "hbf-page-read-queue-depth-per-stack="
        << options.hbf_page_read_queue_depth_per_stack << '\n';
    out << "static-direct-hbm-bytes=" << options.static_direct_hbm_bytes << '\n';
    out << "hbf-gc-low-watermark-pages=" << options.hbf_gc_low_watermark_pages << '\n';
    out << "hbf-gc-hard-watermark-pages=" << options.hbf_gc_hard_watermark_pages << '\n';
    out << "hbf-gc-reserved-free-blocks-per-plane="
        << options.hbf_gc_reserved_free_blocks_per_plane << '\n';
    out << "hbf-gc-wear-leveling-weight=" << options.hbf_gc_wear_leveling_weight << '\n';
    out << "hbf-write-coalescing=" << (options.hbf_write_coalescing ? "true" : "false") << '\n';
    out << "hbf-write-buffer-completion-requires-flush="
        << (options.hbf_write_buffer_completion_requires_flush ? "true" : "false") << '\n';
    out << "hbf-write-buffer-pages=" << options.hbf_write_buffer_pages << '\n';
    out << "hbf-write-buffer-flush-threshold-pages="
        << options.hbf_write_buffer_flush_threshold_pages << '\n';
    out << "hbf-hbm-write-buffer-bytes=" << options.hbf_hbm_write_buffer_bytes << '\n';
    out << "hbf-hbm-write-buffer-destage=" << options.hbf_hbm_write_buffer_destage << '\n';
    out << "trace-mode=" << (options.trace_mode == TraceMode::Off ? "off" :
        options.trace_mode == TraceMode::Summary ? "summary" :
        options.trace_mode == TraceMode::Sampled ? "sampled" : "full") << '\n';
    finish_output(out, output_path, "resolved config");
}

void write_summary_csv(const std::string& output_path, const std::vector<ScenarioResult>& results) {
    ensure_parent_dir(output_path);
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot open summary CSV output: " + output_path);
    }
    const auto& columns = result_columns();
    for (std::size_t i = 0; i < columns.size(); ++i) {
        out << (i == 0 ? "" : ",") << columns[i].header;
    }
    out << '\n';
    for (const auto& result : results) {
        for (std::size_t i = 0; i < columns.size(); ++i) {
            out << (i == 0 ? "" : ",") << columns[i].cell(result).csv;
        }
        out << '\n';
    }
    finish_output(out, output_path, "summary CSV");
}

bool run_sanity_checks(
    std::vector<ScenarioResult>& results,
    const Options& options,
    std::uint64_t trace_footprint_bytes) {
    bool ok = true;
    const auto resolved_hbf = make_hbf_config(options);
    const auto resolved_external_backing = make_external_backing_config(options);
    auto by_name = [&results](const std::string& name) -> ScenarioResult* {
        for (auto& result : results) {
            if (result.name == name) {
                return &result;
            }
        }
        return nullptr;
    };

    for (auto& result : results) {
        if (!result.address_heatmap) {
            result.warnings.push_back("required address heatmap snapshot is missing");
            ok = false;
        } else {
            const auto& heatmap = *result.address_heatmap;
            const auto& workload = heatmap.domains[
                static_cast<std::size_t>(AddressDomain::WorkloadLogical)];
            const auto& hbm_physical = heatmap.domains[
                static_cast<std::size_t>(AddressDomain::HbmPhysical)];
            const auto& hbf_physical = heatmap.domains[
                static_cast<std::size_t>(AddressDomain::HbfPhysical)];
            const auto& external_physical = heatmap.domains[
                static_cast<std::size_t>(AddressDomain::ExternalPhysical)];
            if (heatmap.bin_count != options.address_heatmap_bins ||
                workload.domain != AddressDomain::WorkloadLogical ||
                hbm_physical.domain != AddressDomain::HbmPhysical ||
                hbf_physical.domain != AddressDomain::HbfPhysical ||
                external_physical.domain !=
                    AddressDomain::ExternalPhysical) {
                result.warnings.push_back("address heatmap domain/bin contract mismatch");
                ok = false;
            }
            if (workload.total.read_bytes + workload.total.write_bytes !=
                    result.logical_bytes ||
                workload.total.erase_bytes != 0) {
                result.warnings.push_back(
                    "workload heatmap bytes do not conserve logical request bytes");
                ok = false;
            }
            const auto expected_hbm_read = result.has_hbm ? result.hbm_stats.read_bytes : 0;
            const auto expected_hbm_write = result.has_hbm ? result.hbm_stats.write_bytes : 0;
            if (hbm_physical.total.read_bytes != expected_hbm_read ||
                hbm_physical.total.write_bytes != expected_hbm_write ||
                hbm_physical.total.erase_bytes != 0) {
                result.warnings.push_back(
                    "HBM physical heatmap bytes do not conserve HBM statistics");
                ok = false;
            }
            const auto expected_hbf_read =
                result.has_hbf ? result.hbf_stats.physical_read_bytes : 0;
            const auto expected_hbf_write =
                result.has_hbf ? result.hbf_stats.physical_write_bytes : 0;
            const auto expected_hbf_erase = result.has_hbf ? checked_mul_u64(
                result.hbf_stats.block_erases,
                checked_mul_u64(
                    options.hbf_pages_per_block,
                    options.hbf_page_size,
                    "HBF heatmap block bytes"),
                "HBF heatmap erase bytes") : 0;
            if (hbf_physical.total.read_bytes != expected_hbf_read ||
                hbf_physical.total.write_bytes != expected_hbf_write ||
                hbf_physical.total.erase_bytes != expected_hbf_erase) {
                result.warnings.push_back(
                    "HBF physical heatmap bytes do not conserve HBF statistics");
                ok = false;
            }
            const auto expected_external_read = result.has_external_backing ?
                result.external_backing_stats.read_bytes : 0;
            const auto expected_external_write = result.has_external_backing ?
                result.external_backing_stats.write_bytes : 0;
            if (external_physical.total.read_bytes !=
                    expected_external_read ||
                external_physical.total.write_bytes !=
                    expected_external_write ||
                external_physical.total.erase_bytes != 0) {
                result.warnings.push_back(
                    "external physical heatmap bytes do not conserve "
                    "external-backing statistics");
                ok = false;
            }
        }
        if (result.ops == 0 ||
            result.service_latencies_ns.size() != result.ops ||
            result.offered_latencies_ns.size() != result.ops ||
            result.source_latencies_ns.size() != result.ops) {
            result.warnings.push_back("operation accounting mismatch");
            ok = false;
        }
        for (std::size_t i = 0;
             i < result.service_latencies_ns.size() &&
             i < result.offered_latencies_ns.size() &&
             i < result.source_latencies_ns.size();
             ++i) {
            const auto service = result.service_latencies_ns[i];
            const auto offered = result.offered_latencies_ns[i];
            const auto source = result.source_latencies_ns[i];
            if (!std::isfinite(service) || service < 0.0 ||
                !std::isfinite(offered) || offered < service ||
                !std::isfinite(source) || source < offered) {
                result.warnings.push_back(
                    "non-finite, negative, or non-causal latency");
                ok = false;
                break;
            }
        }
        if (!std::isfinite(result.first_arrival_ns) ||
            !std::isfinite(result.last_arrival_ns) ||
            !std::isfinite(result.user_finish_ns) ||
            !std::isfinite(result.finish_ns) ||
            result.first_arrival_ns < 0.0 ||
            result.last_arrival_ns < result.first_arrival_ns ||
            result.user_finish_ns < result.last_arrival_ns ||
            result.finish_ns < result.user_finish_ns) {
            result.warnings.push_back("invalid scenario timeline ordering");
            ok = false;
        }
        if (result.has_hbm && result.hbm_stats.finish_ns > result.finish_ns + 1e-6) {
            result.warnings.push_back("HBM device finish exceeds scenario finish");
            ok = false;
        }
        if (result.has_hbm &&
            result.hbm_stats.max_queue_occupancy > options.hbm_queue_depth) {
            result.warnings.push_back("HBM queue occupancy exceeded configured depth");
            ok = false;
        }
        if (result.has_hbm) {
            const double utilization = result.hbm_stats.utilization();
            const double parallelism = result.hbm_stats.bus_parallelism();
            const double pseudo_channels =
                static_cast<double>(result.hbm_stats.pseudo_channels);
            if (!std::isfinite(utilization) || utilization < 0.0 ||
                utilization > 1.0 + 1e-9) {
                result.warnings.push_back(
                    "HBM bus utilization falls outside the physical [0, 1] bound");
                ok = false;
            }
            if (!std::isfinite(parallelism) || parallelism < 0.0 ||
                parallelism > pseudo_channels + 1e-9) {
                result.warnings.push_back(
                    "HBM bus parallelism exceeds the pseudo-channel count");
                ok = false;
            }
        }
        if (result.has_hbf && result.hbf_stats.finish_ns > result.finish_ns + 1e-6) {
            result.warnings.push_back("HBF device finish exceeds scenario finish");
            ok = false;
        }
        if (result.has_hbf) {
            const double codeword_bytes = static_cast<double>(
                options.hbf_page_size + options.hbf_oob_bytes);
            const double dies_per_stack = static_cast<double>(
                options.hbf_channels) * options.hbf_dies_per_channel;
            const double payload_fraction =
                static_cast<double>(options.hbf_page_size) / codeword_bytes;
            const double decode_ceiling_GBps = dies_per_stack *
                options.hbf_ecc_decode_raw_bw_GBps_per_die * payload_fraction;
            const double encode_ceiling_GBps = dies_per_stack *
                options.hbf_ecc_encode_raw_bw_GBps_per_die * payload_fraction;
            const double channel_ceiling_GBps =
                static_cast<double>(options.hbf_channels) *
                options.hbf_channel_bw_GBps * payload_fraction;
            const double tsv_wire_bytes = codeword_bytes +
                static_cast<double>(resolved_hbf.command_address_bytes);
            const double tsv_payload_fraction =
                static_cast<double>(options.hbf_page_size) / tsv_wire_bytes;
            const double tsv_ceiling_GBps =
                options.hbf_tsv_bw_GBps * tsv_payload_fraction;
            if (decode_ceiling_GBps + 1e-9 < options.hbf_hbio_bw_GBps) {
                result.warnings.push_back(
                    "configured ECC decode payload ceiling is below HBF HBIO bandwidth");
            }
            if (encode_ceiling_GBps + 1e-9 < options.hbf_hbio_bw_GBps) {
                result.warnings.push_back(
                    "configured ECC encode payload ceiling is below HBF HBIO bandwidth");
            }
            if (channel_ceiling_GBps + 1e-9 < options.hbf_hbio_bw_GBps) {
                result.warnings.push_back(
                    "aggregate raw channel payload ceiling is below HBF HBIO bandwidth");
            }
            if (tsv_ceiling_GBps + 1e-9 < options.hbf_hbio_bw_GBps) {
                result.warnings.push_back(
                    "shared TSV command+raw payload ceiling is below HBF HBIO bandwidth");
            }
        }
        const auto hbf_served_read_bytes = result.has_hbf ?
            checked_add_u64(
                checked_add_u64(
                    result.hbf_stats.physical_read_bytes,
                    result.hbf_stats.write_buffer_read_bytes,
                    "HBF served read byte count"),
                result.hbf_stats.read_buffer_read_bytes,
                "HBF served read byte count") : 0;
        if (result.has_hbf &&
            hbf_served_read_bytes < result.hbf_stats.logical_read_bytes) {
            result.warnings.push_back(
                "HBF served read bytes below logical read bytes");
            ok = false;
        }
        if (result.has_hbf && !options.hbf_write_coalescing &&
            result.hbf_stats.physical_write_bytes < result.hbf_stats.logical_write_bytes) {
            result.warnings.push_back("HBF physical write bytes below logical write bytes");
            ok = false;
        }
        if (result.has_external_backing) {
            const auto& stats = result.external_backing_stats;
            if (stats.finish_ns > result.finish_ns + 1e-6) {
                result.warnings.push_back(
                    "external-backing finish exceeds scenario finish");
                ok = false;
            }
            if (stats.read_requests + stats.write_requests !=
                    result.external_accesses ||
                stats.max_device_outstanding >
                    resolved_external_backing.max_outstanding_requests ||
                stats.m2s_payload_bytes > stats.m2s_wire_bytes ||
                stats.m2s_protocol_bytes !=
                    stats.m2s_wire_bytes - stats.m2s_payload_bytes ||
                stats.s2m_payload_bytes > stats.s2m_wire_bytes ||
                stats.s2m_protocol_bytes !=
                    stats.s2m_wire_bytes - stats.s2m_payload_bytes ||
                stats.m2s_payload_bytes != stats.write_bytes ||
                stats.s2m_payload_bytes != stats.read_bytes) {
                result.warnings.push_back(
                    "external-backing request/queue accounting is inconsistent");
                ok = false;
            }
            const std::array<double, 4> utilization{
                stats.controller_utilization(),
                stats.media_utilization(),
                stats.m2s_utilization(),
                stats.s2m_utilization(),
            };
            if (std::any_of(
                    utilization.begin(),
                    utilization.end(),
                    [](double value) {
                        return !std::isfinite(value) || value < 0.0 ||
                            value > 1.0 + 1e-9;
                    })) {
                result.warnings.push_back(
                    "external-backing utilization falls outside [0, 1]");
                ok = false;
            }
        }
        for (const auto& warning : result.warnings) {
            if (warning.find("unmapped") != std::string::npos ||
                warning.find("accounting mismatch") != std::string::npos ||
                warning.find("did not exercise both") != std::string::npos) {
                ok = false;
            }
        }
    }

    if (auto* all_hbm = by_name(kAllHbmScenario)) {
        if (all_hbm->hbm_user_accesses < all_hbm->ops ||
            all_hbm->hbm_accesses != all_hbm->hbm_user_accesses ||
            all_hbm->hbf_accesses != 0) {
            all_hbm->warnings.push_back("all-HBM routed outside HBM");
            ok = false;
        }
        if (trace_footprint_bytes > options.hbm_capacity_bytes) {
            all_hbm->warnings.push_back(
                "trace unique-line footprint exceeds HBM capacity; all-HBM is capacity-infeasible");
            ok = false;
        }
        // Policy identity: a FLAT run that routed NOTHING to HBF saw the
        // exact same op stream over the exact same HBM path as all-HBM, so
        // the two results must be bit-equal (a divergence is a routing or
        // driver bug, not a modeling choice). Structural form of the
        // comparison-table observation that FLAT degenerates to all-HBM
        // when the split covers the footprint.
        if (auto* flat = by_name(kFlatScenario)) {
            if (flat->hbf_accesses == 0 && flat->hbf_user_accesses == 0) {
                if (flat->finish_ns != all_hbm->finish_ns ||
                    flat->hbm_accesses != all_hbm->hbm_accesses ||
                    flat->service_latencies_ns !=
                        all_hbm->service_latencies_ns ||
                    flat->offered_latencies_ns !=
                        all_hbm->offered_latencies_ns ||
                    flat->source_latencies_ns !=
                        all_hbm->source_latencies_ns) {
                    flat->warnings.push_back(
                        "policy identity violated: FLAT with no HBF traffic "
                        "must equal all-HBM bit for bit");
                    ok = false;
                }
            }
        }
    }
    if (auto* all_hbf = by_name(kAllHbfScenario)) {
        if (all_hbf->hbf_user_accesses < all_hbf->ops ||
            all_hbf->hbm_accesses != 0) {
            all_hbf->warnings.push_back("all-HBF routed outside HBF");
            ok = false;
        }
        if (all_hbf->writes > 0 && !options.hbf_write_coalescing) {
            const auto waf = all_hbf->hbf_stats.waf();
            if (waf && *waf < 1.0) {
                all_hbf->warnings.push_back(
                    "HBF WAF below 1.0 is physically suspicious");
                ok = false;
            }
        }
    }
    const auto check_behavioral = [&](
        const char* scenario_name,
        BehavioralAdmissionPolicy expected_policy) {
        auto* behavioral = by_name(scenario_name);
        if (behavioral == nullptr) {
            return;
        }
        if (!behavioral->behavioral_tiering_stats) {
            behavioral->warnings.push_back(
                "behavioral tiering result is missing controller statistics");
            ok = false;
            return;
        }
        const auto& s = *behavioral->behavioral_tiering_stats;
        const auto classified = checked_add_u64(
            checked_add_u64(
                s.hbm_hits,
                s.hbf_bypasses,
                "behavioral classified observations"),
            s.promotions,
            "behavioral classified observations");
        const auto promotion_classes = checked_add_u64(
            s.promotions_with_backing_fill,
            s.promotions_without_backing_fill,
            "behavioral promotion classes");
        const auto expected_fill_bytes = checked_mul_u64(
            s.backing_fill_pages,
            options.hbf_page_size,
            "behavioral fill bytes");
        const auto expected_install_bytes = checked_mul_u64(
            s.hbm_install_pages,
            options.hbf_page_size,
            "behavioral install bytes");
        const auto expected_writeback_bytes = checked_mul_u64(
            s.dirty_writeback_pages,
            options.hbf_page_size,
            "behavioral writeback bytes");
        if (s.admission_policy != expected_policy ||
            s.semantic_inputs_consumed ||
            s.page_observations != classified ||
            s.promotions != promotion_classes ||
            s.promotions_with_backing_fill != s.backing_fill_pages ||
            s.backing_fill_pages != s.hbm_install_pages ||
            s.backing_fill_bytes != expected_fill_bytes ||
            s.hbm_install_bytes != expected_install_bytes ||
            s.dirty_writeback_bytes != expected_writeback_bytes ||
            s.peak_resident_pages > s.hbm_tier_pages ||
            s.final_resident_pages > s.hbm_tier_pages ||
            s.final_dirty_pages != 0 ||
            behavioral->hbf_user_accesses != s.hbf_bypasses ||
            behavioral->hbm_user_accesses < s.hbm_hits ||
            behavioral->base_die_link_stats.read_bytes !=
                s.backing_fill_bytes ||
            behavioral->base_die_link_stats.write_bytes !=
                s.dirty_writeback_bytes ||
            behavioral->background_hbf_writes !=
                s.dirty_writeback_pages ||
            behavioral->hbf_backing_write_bytes !=
                s.dirty_writeback_bytes ||
            behavioral->hbm_foreground_bytes !=
                s.hbm_foreground_bytes) {
            behavioral->warnings.push_back(
                "behavioral placement conservation or policy contract failed");
            ok = false;
        }
    };
    check_behavioral(
        kDemandFillScenario,
        BehavioralAdmissionPolicy::AlwaysAdmit);
    check_behavioral(
        kBehavioralTieringScenario,
        BehavioralAdmissionPolicy::ReuseFiltered);

    const auto check_streaming = [&](
        const char* scenario_name,
        BackingTier expected_backing,
        std::uint64_t page_size) {
        auto* streaming = by_name(scenario_name);
        if (streaming == nullptr) {
            return;
        }
        if (!streaming->layer_streaming_stats) {
            streaming->warnings.push_back(
                "layer-streaming result is missing controller statistics");
            ok = false;
        } else {
            const auto& s = *streaming->layer_streaming_stats;
            const auto foreground_page_transactions = checked_add_u64(
                s.foreground_resident_page_accesses,
                s.foreground_buffer_page_accesses,
                "layer-streaming foreground page transactions");
            if (streaming->hbm_user_accesses !=
                    foreground_page_transactions ||
                streaming->hbm_user_accesses < streaming->ops ||
                streaming->hbf_user_accesses != 0 ||
                s.backing != expected_backing) {
                streaming->warnings.push_back(
                    "layer streaming must execute every foreground page "
                    "transaction from HBM with the selected backing tier");
                ok = false;
            }
            const std::array<double, 8> times{
                s.compute_work_ns,
                s.backing_admission_wait_work_ns,
                s.backing_admission_max_wait_ns,
                s.user_wait_work_ns,
                s.user_max_wait_ns,
                s.exposed_prefetch_ns,
                s.hidden_prefetch_ns,
                s.buffer_reuse_wait_work_ns,
            };
            if (std::any_of(times.begin(), times.end(), [](double value) {
                    return !std::isfinite(value) || value < 0.0;
                })) {
                streaming->warnings.push_back(
                    "layer-streaming controller reported invalid time work");
                ok = false;
            }
            if (s.streamed_bytes != checked_mul_u64(
                    s.streamed_pages,
                    page_size,
                    "layer-streaming DMA bytes") ||
                s.writeback_bytes != checked_mul_u64(
                    s.dirty_pages_written_back,
                    page_size,
                    "layer-streaming writeback bytes") ||
                s.hbm_only_resident_bytes != checked_mul_u64(
                    s.hbm_only_resident_pages,
                    page_size,
                    "layer-streaming HBM-only resident bytes") ||
                s.hot_kv_candidate_bytes != checked_mul_u64(
                    s.hot_kv_candidate_pages,
                    page_size,
                    "layer-streaming hot-KV candidate bytes") ||
                s.hot_kv_resident_bytes != checked_mul_u64(
                    s.hot_kv_resident_pages,
                    page_size,
                    "layer-streaming hot-KV resident bytes") ||
                s.data_bytes != checked_mul_u64(
                    s.data_pages,
                    page_size,
                    "layer-streaming data bytes") ||
                s.model_weight_resident_bytes != checked_mul_u64(
                    s.model_weight_resident_pages,
                    page_size,
                    "layer-streaming model-weight resident bytes") ||
                s.model_weight_backing_bytes != checked_mul_u64(
                    s.model_weight_backing_pages,
                    page_size,
                    "layer-streaming model-weight backing bytes") ||
                s.cold_kv_backing_bytes != checked_mul_u64(
                    s.cold_kv_backing_pages,
                    page_size,
                    "layer-streaming cold-KV backing bytes") ||
                s.unknown_backing_bytes != checked_mul_u64(
                    s.unknown_backing_pages,
                    page_size,
                    "layer-streaming unknown backing bytes") ||
                s.backing_unique_bytes != checked_mul_u64(
                    s.backing_unique_pages,
                    page_size,
                    "layer-streaming unique backing bytes") ||
                s.unique_resident_footprint_bytes != checked_mul_u64(
                    s.unique_resident_footprint_pages,
                    page_size,
                    "layer-streaming unique footprint bytes") ||
                s.capacity_pressure_basis_bytes >
                    s.unique_resident_footprint_bytes ||
                s.footprint_page_rounding_bytes !=
                    s.unique_resident_footprint_bytes -
                        s.capacity_pressure_basis_bytes ||
                s.resident_physical_bytes != checked_mul_u64(
                    s.resident_physical_pages,
                    page_size,
                    "layer-streaming resident physical bytes") ||
                s.effective_layer_buffer_bytes != checked_mul_u64(
                    s.effective_layer_buffer_pages,
                    page_size,
                    "layer-streaming effective buffer bytes") ||
                s.max_layer_data_bytes != checked_mul_u64(
                    s.max_layer_data_pages,
                    page_size,
                    "layer-streaming maximum layer data bytes") ||
                s.unused_hbm_bytes != checked_mul_u64(
                    s.unused_hbm_pages,
                    page_size,
                    "layer-streaming unused HBM bytes") ||
                s.hot_kv_resident_pages > s.hot_kv_candidate_pages ||
                s.backing_unique_pages != checked_add_u64(
                    s.model_weight_backing_pages,
                    checked_add_u64(
                        s.cold_kv_backing_pages,
                        s.unknown_backing_pages,
                        "layer-streaming cold-KV and unknown pages"),
                    "layer-streaming backing partition") ||
                s.data_pages != checked_add_u64(
                    checked_add_u64(
                        s.model_weight_resident_pages,
                        s.hot_kv_resident_pages,
                        "layer-streaming resident data pages"),
                    s.backing_unique_pages,
                    "layer-streaming data partition") ||
                s.unique_resident_footprint_pages != checked_add_u64(
                    s.hbm_only_resident_pages,
                    s.data_pages,
                    "layer-streaming unique footprint") ||
                s.resident_physical_pages != checked_add_u64(
                    s.hbm_only_resident_pages,
                    checked_add_u64(
                        s.model_weight_resident_pages,
                        s.hot_kv_resident_pages,
                        "layer-streaming resident model weights and hot KV"),
                    "layer-streaming HBM residency") ||
                (!s.explicit_residency_contract &&
                 s.effective_layer_buffer_pages !=
                    s.max_layer_data_pages) ||
                (s.explicit_residency_contract &&
                 s.effective_layer_buffer_pages <
                    s.max_layer_data_pages) ||
                s.max_layer_data_bytes > options.layer_buffer_bytes) {
                streaming->warnings.push_back(
                    "layer-streaming page/byte/capacity accounting is inconsistent");
                ok = false;
            }
            const auto backing_transactions = checked_add_u64(
                s.streamed_pages,
                s.dirty_pages_written_back,
                "layer-streaming backing transactions");
            if (s.backing_request_credit_limit !=
                    options.max_outstanding_requests ||
                s.backing_max_inflight_requests > backing_transactions ||
                (s.backing_request_credit_limit != 0 &&
                 s.backing_max_inflight_requests >
                    s.backing_request_credit_limit) ||
                s.backing_admission_waited_requests >
                    backing_transactions ||
                ((s.backing_admission_waited_requests == 0) !=
                 (s.backing_admission_wait_work_ns == 0.0)) ||
                ((s.backing_admission_waited_requests == 0) !=
                 (s.backing_admission_max_wait_ns == 0.0)) ||
                s.backing_admission_max_wait_ns >
                    s.backing_admission_wait_work_ns ||
                (s.backing_request_credit_limit == 0 &&
                 s.backing_admission_waited_requests != 0)) {
                streaming->warnings.push_back(
                    "layer-streaming backing credit accounting is "
                    "inconsistent");
                ok = false;
            }
            const auto runtime_hbm_pages = checked_add_u64(
                s.resident_physical_pages,
                checked_mul_u64(
                    s.effective_layer_buffer_pages,
                    2,
                    "two effective layer buffers"),
                "runtime layer-streaming HBM pages");
            const auto expected_capacity_pressure =
                static_cast<double>(
                    static_cast<long double>(
                        s.capacity_pressure_basis_bytes) /
                    static_cast<long double>(s.hbm_capacity_bytes));
            const auto pressure_tolerance =
                std::numeric_limits<double>::epsilon() *
                std::max(1.0, expected_capacity_pressure) * 8.0;
            if (s.hbm_capacity_bytes != options.hbm_capacity_bytes ||
                checked_mul_u64(
                    runtime_hbm_pages,
                    page_size,
                    "runtime layer-streaming HBM bytes") >
                    s.hbm_capacity_bytes ||
                s.compact_resident_mapping !=
                    (s.resident_physical_pages != 0) ||
                checked_add_u64(
                    runtime_hbm_pages,
                    s.unused_hbm_pages,
                    "allocated and unused HBM pages") !=
                    s.hbm_capacity_bytes / page_size ||
                !std::isfinite(s.hbm_capacity_pressure) ||
                s.hbm_capacity_pressure < 0.0 ||
                std::abs(
                    s.hbm_capacity_pressure -
                    expected_capacity_pressure) > pressure_tolerance) {
                streaming->warnings.push_back(
                    "layer-streaming hybrid-residency policy is inconsistent "
                    "with its unique footprint and HBM capacity");
                ok = false;
            }
            if (s.explicit_residency_contract !=
                    options.explicit_residency_contract) {
                streaming->warnings.push_back(
                    "layer-streaming residency contract selection drifted "
                    "from the resolved configuration");
                ok = false;
            }
            if (s.explicit_residency_contract &&
                options.explicit_residency_contract) {
                const auto logical_kv_bytes = checked_mul_u64(
                    s.logical_kv_blocks,
                    s.kv_block_stride_bytes,
                    "explicit logical KV bytes");
                const auto exact_population = checked_add_u64(
                    s.immutable_weight_logical_bytes,
                    checked_add_u64(
                        s.runtime_overhead_logical_bytes,
                        checked_add_u64(
                            s.block_table_logical_bytes,
                            logical_kv_bytes,
                            "explicit block table and logical KV"),
                        "explicit overhead, block table, and logical KV"),
                    "explicit unique resident population");
                if (s.capacity_pressure_basis_bytes != exact_population ||
                    s.logical_kv_blocks != checked_add_u64(
                        s.hot_kv_blocks,
                        s.cold_kv_blocks,
                        "explicit hot and cold KV blocks") ||
                    s.immutable_weight_logical_bytes !=
                        *options.residency_immutable_weight_bytes ||
                    s.runtime_overhead_logical_bytes !=
                        *options.residency_runtime_overhead_bytes ||
                    s.block_table_logical_bytes !=
                        *options.residency_block_table_bytes ||
                    s.active_buffer_logical_bytes_per_slot !=
                        *options.residency_active_buffer_bytes ||
                    s.residency_page_size_bytes !=
                        *options.residency_page_size_bytes ||
                    s.kv_block_stride_bytes !=
                        *options.residency_kv_block_stride_bytes ||
                    s.logical_kv_blocks !=
                        *options.residency_logical_kv_blocks ||
                    s.hot_kv_blocks !=
                        *options.residency_hot_kv_blocks ||
                    s.model_weight_resident_pages !=
                        *options.residency_static_weight_pages) {
                    streaming->warnings.push_back(
                        "explicit layer-streaming residency contract is not "
                        "byte-conservative");
                    ok = false;
                }
            } else if (!s.explicit_residency_contract && (
                s.capacity_pressure_basis_bytes !=
                    s.unique_resident_footprint_bytes ||
                s.footprint_page_rounding_bytes != 0 ||
                s.immutable_weight_logical_bytes != 0 ||
                s.runtime_overhead_logical_bytes != 0 ||
                s.block_table_logical_bytes != 0 ||
                s.active_buffer_logical_bytes_per_slot != 0 ||
                s.residency_page_size_bytes != 0 ||
                s.kv_block_stride_bytes != 0 ||
                s.logical_kv_blocks != 0 ||
                s.hot_kv_blocks != 0 ||
                s.cold_kv_blocks != 0)) {
                streaming->warnings.push_back(
                    "trace-derived layer streaming leaked an explicit "
                    "residency contract");
                ok = false;
            }
            if (s.resident_physical_bytes +
                    checked_mul_u64(
                        s.effective_layer_buffer_bytes,
                        2,
                        "two runtime layer buffers") >
                s.hbm_capacity_bytes) {
                streaming->warnings.push_back(
                    "layer-streaming runtime allocation exceeds HBM");
                ok = false;
            }
            if (expected_backing == BackingTier::Hbf) {
                if (s.streamed_bytes != checked_add_u64(
                        streaming->hbf_static_read_bytes,
                        streaming->hbf_logical_read_bytes,
                        "HBF backing read bytes") ||
                    s.writeback_bytes !=
                        streaming->hbf_backing_write_bytes ||
                    streaming->external_accesses != 0) {
                    streaming->warnings.push_back(
                        "HBF layer-streaming backing traffic does not conserve");
                    ok = false;
                }
            } else if (
                s.streamed_bytes != streaming->external_backing_read_bytes ||
                s.writeback_bytes !=
                    streaming->external_backing_write_bytes ||
                streaming->hbf_accesses != 0 ||
                streaming->base_die_link_stats.read_transfers != 0 ||
                streaming->base_die_link_stats.write_transfers != 0) {
                streaming->warnings.push_back(
                    "external layer-streaming backing traffic does not conserve");
                ok = false;
            }
        }
    };
    check_streaming(
        kLayerStreamingScenario,
        BackingTier::Hbf,
        options.hbf_page_size);
    check_streaming(
        kExternalLayerStreamingScenario,
        BackingTier::External,
        resolved_external_backing.page_size_bytes);

    if (options.hbf_page_size == resolved_external_backing.page_size_bytes) {
        auto* hbf_streaming = by_name(kLayerStreamingScenario);
        auto* external_streaming =
            by_name(kExternalLayerStreamingScenario);
        if (hbf_streaming != nullptr && external_streaming != nullptr &&
            hbf_streaming->layer_streaming_stats &&
            external_streaming->layer_streaming_stats) {
            const auto& hbf_plan =
                *hbf_streaming->layer_streaming_stats;
            const auto& external_plan =
                *external_streaming->layer_streaming_stats;
            const bool same_plan =
                hbf_plan.compact_resident_mapping ==
                    external_plan.compact_resident_mapping &&
                hbf_plan.semantic_inputs_consumed ==
                    external_plan.semantic_inputs_consumed &&
                hbf_plan.explicit_residency_contract ==
                    external_plan.explicit_residency_contract &&
                hbf_plan.address_footprint_bytes ==
                    external_plan.address_footprint_bytes &&
                hbf_plan.unique_resident_footprint_pages ==
                    external_plan.unique_resident_footprint_pages &&
                hbf_plan.unique_resident_footprint_bytes ==
                    external_plan.unique_resident_footprint_bytes &&
                hbf_plan.capacity_pressure_basis_bytes ==
                    external_plan.capacity_pressure_basis_bytes &&
                hbf_plan.footprint_page_rounding_bytes ==
                    external_plan.footprint_page_rounding_bytes &&
                hbf_plan.hbm_capacity_pressure ==
                    external_plan.hbm_capacity_pressure &&
                hbf_plan.layers == external_plan.layers &&
                hbf_plan.explicit_layer_requests ==
                    external_plan.explicit_layer_requests &&
                hbf_plan.explicit_compute_layers ==
                    external_plan.explicit_compute_layers &&
                hbf_plan.compute_work_ns ==
                    external_plan.compute_work_ns &&
                hbf_plan.hbm_only_resident_pages ==
                    external_plan.hbm_only_resident_pages &&
                hbf_plan.hot_kv_candidate_pages ==
                    external_plan.hot_kv_candidate_pages &&
                hbf_plan.hot_kv_resident_pages ==
                    external_plan.hot_kv_resident_pages &&
                hbf_plan.data_pages == external_plan.data_pages &&
                hbf_plan.model_weight_resident_pages ==
                    external_plan.model_weight_resident_pages &&
                hbf_plan.model_weight_backing_pages ==
                    external_plan.model_weight_backing_pages &&
                hbf_plan.cold_kv_backing_pages ==
                    external_plan.cold_kv_backing_pages &&
                hbf_plan.unknown_backing_pages ==
                    external_plan.unknown_backing_pages &&
                hbf_plan.backing_unique_pages ==
                    external_plan.backing_unique_pages &&
                hbf_plan.resident_physical_pages ==
                    external_plan.resident_physical_pages &&
                hbf_plan.effective_layer_buffer_pages ==
                    external_plan.effective_layer_buffer_pages &&
                hbf_plan.unused_hbm_pages ==
                    external_plan.unused_hbm_pages &&
                hbf_plan.streamed_pages ==
                    external_plan.streamed_pages &&
                hbf_plan.foreground_resident_page_accesses ==
                    external_plan.foreground_resident_page_accesses &&
                hbf_plan.foreground_buffer_page_accesses ==
                    external_plan.foreground_buffer_page_accesses &&
                hbf_plan.dirty_pages_written_back ==
                    external_plan.dirty_pages_written_back &&
                hbf_plan.max_layer_data_pages ==
                    external_plan.max_layer_data_pages &&
                hbf_plan.immutable_weight_logical_bytes ==
                    external_plan.immutable_weight_logical_bytes &&
                hbf_plan.runtime_overhead_logical_bytes ==
                    external_plan.runtime_overhead_logical_bytes &&
                hbf_plan.block_table_logical_bytes ==
                    external_plan.block_table_logical_bytes &&
                hbf_plan.active_buffer_logical_bytes_per_slot ==
                    external_plan.active_buffer_logical_bytes_per_slot &&
                hbf_plan.residency_page_size_bytes ==
                    external_plan.residency_page_size_bytes &&
                hbf_plan.kv_block_stride_bytes ==
                    external_plan.kv_block_stride_bytes &&
                hbf_plan.logical_kv_blocks ==
                    external_plan.logical_kv_blocks &&
                hbf_plan.hot_kv_blocks ==
                    external_plan.hot_kv_blocks &&
                hbf_plan.cold_kv_blocks ==
                    external_plan.cold_kv_blocks &&
                hbf_streaming->hbm_user_accesses ==
                    external_streaming->hbm_user_accesses &&
                hbf_streaming->hbm_background_accesses ==
                    external_streaming->hbm_background_accesses &&
                hbf_streaming->hbm_foreground_bytes ==
                    external_streaming->hbm_foreground_bytes &&
                hbf_streaming->hbm_streaming_write_bytes ==
                    external_streaming->hbm_streaming_write_bytes;
            if (!same_plan) {
                external_streaming->warnings.push_back(
                    "HBF and external layer-streaming controls diverged "
                    "before the backing-device boundary");
                ok = false;
            }
        }
    }

    return ok;
}

std::string json_escape(const std::string& input) {
    std::ostringstream out;
    for (const char ch : input) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(static_cast<unsigned char>(ch))
                    << std::dec;
            } else {
                out << ch;
            }
            break;
        }
    }
    return out.str();
}

void write_stage_work_json(
    std::ostream& out,
    const Breakdown& work,
    const std::string& indent,
    const hbfsim::physical::hbf::HbfStats* hbf_stats = nullptr) {
    out << "{\n"
        << indent << "  \"ingress_queue_wait_work_ns\": "
        << work.ingress_queue_wait_ns << ",\n"
        << indent << "  \"scheduler_queue_wait_work_ns\": "
        << work.scheduler_queue_wait_ns << ",\n"
        << indent << "  \"address_mapping_work_ns\": "
        << work.address_mapping_ns << ",\n"
        << indent << "  \"translation_work_ns\": " << work.translation_ns << ",\n"
        << indent << "  \"mapping_dram_work_ns\": "
        << work.mapping_dram_ns << ",\n"
        << indent << "  \"refresh_stall_work_ns\": "
        << work.refresh_stall_ns << ",\n"
        << indent << "  \"precharge_work_ns\": " << work.precharge_ns << ",\n"
        << indent << "  \"activation_work_ns\": " << work.activation_ns << ",\n"
        << indent << "  \"command_work_ns\": " << work.command_ns << ",\n"
        << indent << "  \"array_read_work_ns\": " << work.array_read_ns << ",\n"
        << indent << "  \"array_program_work_ns\": "
        << work.array_program_ns << ",\n"
        << indent << "  \"array_erase_work_ns\": " << work.array_erase_ns << ",\n"
        << indent << "  \"program_verify_work_ns\": "
        << work.program_verify_ns << ",\n"
        << indent << "  \"media_lane_transfer_work_ns\": "
        << work.media_lane_transfer_ns << ",\n"
        << indent << "  \"page_buffer_work_ns\": " << work.page_buffer_ns << ",\n"
        << indent << "  \"sram_staging_work_ns\": "
        << work.sram_staging_ns << ",\n"
        << indent << "  \"channel_transfer_work_ns\": "
        << work.channel_transfer_ns << ",\n"
        << indent << "  \"tsv_transfer_work_ns\": "
        << work.tsv_transfer_ns << ",\n"
        << indent << "  \"hb_io_transfer_work_ns\": "
        << work.hb_io_transfer_ns << ",\n"
        << indent << "  \"transport_latency_work_ns\": "
        << work.transport_latency_ns << ",\n"
        << indent << "  \"ecc_queue_wait_work_ns\": "
        << work.ecc_queue_wait_ns << ",\n"
        << indent << "  \"ecc_response_latency_work_ns\": "
        << work.ecc_latency_ns << ",\n"
        << indent << "  \"maintenance_work_ns\": " << work.maintenance_ns << ",\n"
        << indent << "  \"total_overlapping_work_ns\": "
        << work.total_work_ns();
    if (hbf_stats != nullptr) {
        const auto& stats = *hbf_stats;
        out << ",\n"
            << indent << "  \"write_buffer\": {"
            << "\"slot_waited_ops\": " << stats.write_buffer_slot_wait_ops
            << ", \"slot_wait_work_ns\": "
            << stats.write_buffer_slot_wait_ns << "},\n"
            << indent << "  \"resident_mapping\": {"
            << "\"table_bytes\": "
            << stats.resident_mapping_table_bytes
            << ", \"table_bytes_per_stack\": "
            << stats.resident_mapping_table_bytes_per_stack
            << ", \"pages_per_stack\": "
            << stats.resident_mapping_pages_per_stack
            << ", \"lookup_ops\": " << stats.mapping_lookup_ops
            << ", \"user_lookup_ops\": " << stats.mapping_user_lookup_ops
            << ", \"gc_lookup_ops\": " << stats.mapping_gc_lookup_ops
            << ", \"update_ops\": " << stats.mapping_update_ops
            << ", \"user_update_ops\": " << stats.mapping_user_update_ops
            << ", \"gc_update_ops\": " << stats.mapping_gc_update_ops
            << ", \"dram_waited_ops\": " << stats.mapping_dram_wait_ops
            << ", \"dram_wait_work_ns\": " << stats.mapping_dram_wait_ns
            << ", \"dram_wait_max_ns\": " << stats.mapping_dram_wait_max_ns
            << ", \"dram_issue_busy_ns\": "
            << stats.mapping_dram_issue_busy_ns
            << ", \"dram_resources\": " << stats.mapping_dram_resources
            << "},\n"
            << indent << "  \"ecc_directional\": {\n"
            << indent << "    \"decode\": {"
            << "\"queue_wait_work_ns\": " << stats.ecc_decode_queue_wait_ns
            << ", \"response_latency_work_ns\": "
            << stats.ecc_decode_latency_work_ns
            << ", \"issue_busy_ns\": " << stats.ecc_decode_issue_busy_ns
            << ", \"ops\": " << stats.ecc_decode_ops
            << ", \"codeword_bytes\": " << stats.ecc_decode_codeword_bytes
            << "},\n"
            << indent << "    \"encode\": {"
            << "\"queue_wait_work_ns\": " << stats.ecc_encode_queue_wait_ns
            << ", \"response_latency_work_ns\": "
            << stats.ecc_encode_latency_work_ns
            << ", \"issue_busy_ns\": " << stats.ecc_encode_issue_busy_ns
            << ", \"ops\": " << stats.ecc_encode_ops
            << ", \"codeword_bytes\": " << stats.ecc_encode_codeword_bytes
            << "}\n"
            << indent << "  }";
    }
    out << '\n' << indent << '}';
}

void write_resource_busy_json(
    std::ostream& out,
    const ResourceBusyMetric& metric) {
    out << "{\"busy_ns\": " << metric.busy_ns
        << ", \"resource_count\": " << metric.resource_count
        << ", \"active_span_ns\": " << metric.active_span_ns
        << ", \"capacity_time_ns\": " << metric.capacity_time_ns
        << ", \"utilization\": " << metric.utilization << '}';
}

void write_time_breakdown_json(
    std::ostream& out,
    const ScenarioResult& result,
    const std::string& indent) {
    const auto time = make_time_breakdown(result);
    const auto& wall = time.wall_clock;
    const auto& latency = time.latency_work;
    out << indent << "\"time_breakdown\": {\n"
        << indent << "  \"contract_version\": 2,\n"
        << indent << "  \"semantics\": {\n"
        << indent << "    \"wall_clock\": \"additive_non_overlapping_spans\",\n"
        << indent << "    \"latency_work\": \"sum_across_user_operations_may_overlap\",\n"
        << indent << "    \"stage_work\": \"hierarchical_device_work_may_overlap_do_not_sum_parent_and_children\",\n"
        << indent << "    \"controller_work\": \"parent_totals_and_attribution_views_may_overlap\",\n"
        << indent << "    \"resource_busy\": \"exclusive_per_resource_then_aggregated\"\n"
        << indent << "  },\n"
        << indent << "  \"wall_clock_ns\": {\n"
        << indent << "    \"trace_origin_ns\": " << wall.trace_origin_ns << ",\n"
        << indent << "    \"last_offered_arrival_ns\": "
        << wall.last_offered_arrival_ns << ",\n"
        << indent << "    \"last_user_completion_ns\": "
        << wall.last_user_completion_ns << ",\n"
        << indent << "    \"quiescent_finish_ns\": "
        << wall.quiescent_finish_ns << ",\n"
        << indent << "    \"offered_arrival_span_ns\": "
        << wall.offered_arrival_span_ns << ",\n"
        << indent << "    \"post_offer_user_completion_tail_ns\": "
        << wall.post_offer_user_completion_tail_ns << ",\n"
        << indent << "    \"user_completion_span_ns\": "
        << wall.user_completion_span_ns << ",\n"
        << indent << "    \"drain_tail_ns\": " << wall.drain_tail_ns << ",\n"
        << indent << "    \"makespan_ns\": " << wall.makespan_ns << "\n"
        << indent << "  },\n"
        << indent << "  \"latency_work\": {\n"
        << indent << "    \"basis\": \"offered_to_user_completion\",\n"
        << indent << "    \"user_count\": " << latency.user_count << ",\n"
        << indent << "    \"service_to_user_completion_sum_work_ns\": "
        << latency.service_to_user_completion_sum_work_ns << ",\n"
        << indent << "    \"phase_barriers\": "
        << latency.phase_barriers << ",\n"
        << indent << "    \"phase_dependency_waited_ops\": "
        << latency.phase_dependency_waited_ops << ",\n"
        << indent << "    \"phase_dependency_wait_work_ns\": "
        << latency.phase_dependency_wait_work_ns << ",\n"
        << indent << "    \"phase_dependency_max_wait_ns\": "
        << latency.phase_dependency_max_wait_ns << ",\n"
        << indent << "    \"front_end_admission_waited_ops\": "
        << latency.front_end_admission_waited_ops << ",\n"
        << indent << "    \"front_end_admission_wait_work_ns\": "
        << latency.front_end_admission_wait_work_ns << ",\n"
        << indent << "    \"front_end_admission_max_wait_ns\": "
        << latency.front_end_admission_max_wait_ns << ",\n"
        << indent << "    \"offered_to_user_completion_sum_work_ns\": "
        << latency.offered_to_user_completion_sum_work_ns << ",\n"
        << indent << "    \"source_to_user_completion_sum_work_ns\": "
        << latency.source_to_user_completion_sum_work_ns << ",\n"
        << indent << "    \"average_ns\": " << latency.average_ns << ",\n"
        << indent << "    \"p50_ns\": " << latency.p50_ns << ",\n"
        << indent << "    \"p95_ns\": " << latency.p95_ns << ",\n"
        << indent << "    \"max_ns\": " << latency.max_ns << ",\n"
        << indent << "    \"service_average_ns\": "
        << latency.service_average_ns << ",\n"
        << indent << "    \"service_p50_ns\": "
        << latency.service_p50_ns << ",\n"
        << indent << "    \"service_p95_ns\": "
        << latency.service_p95_ns << ",\n"
        << indent << "    \"service_max_ns\": "
        << latency.service_max_ns << ",\n"
        << indent << "    \"source_average_ns\": "
        << latency.source_average_ns << ",\n"
        << indent << "    \"source_p50_ns\": "
        << latency.source_p50_ns << ",\n"
        << indent << "    \"source_p95_ns\": "
        << latency.source_p95_ns << ",\n"
        << indent << "    \"source_max_ns\": "
        << latency.source_max_ns << "\n"
        << indent << "  },\n"
        << indent << "  \"stage_work\": {\n"
        << indent << "    \"scope\": \"all_device_work_including_background_and_drain\",\n"
        << indent << "    \"hbm\": ";
    if (result.has_hbm) {
        write_stage_work_json(out, time.hbm_stage_work, indent + "    ");
    } else {
        out << "null";
    }
    out << ",\n" << indent << "    \"hbf\": ";
    if (result.has_hbf) {
        write_stage_work_json(
            out,
            time.hbf_stage_work,
            indent + "    ",
            &result.hbf_stats);
    } else {
        out << "null";
    }
    out << ",\n" << indent << "    \"external_backing\": ";
    if (result.has_external_backing) {
        write_stage_work_json(
            out,
            time.external_backing_stage_work,
            indent + "    ");
    } else {
        out << "null";
    }
    const auto& link = time.base_die_link_stage_work;
    const auto& streaming_controller =
        time.layer_streaming_controller_stage_work;
    const auto& cooperative_controller =
        time.cooperative_write_controller_stage_work;
    out << ",\n"
        << indent << "    \"base_die_link\": {\n"
        << indent << "      \"read_queue_wait_work_ns\": "
        << link.read_queue_wait_work_ns << ",\n"
        << indent << "      \"write_queue_wait_work_ns\": "
        << link.write_queue_wait_work_ns << ",\n"
        << indent << "      \"read_serialization_work_ns\": "
        << link.read_serialization_work_ns << ",\n"
        << indent << "      \"write_serialization_work_ns\": "
        << link.write_serialization_work_ns << ",\n"
        << indent << "      \"read_fixed_latency_work_ns\": "
        << link.read_fixed_latency_work_ns << ",\n"
        << indent << "      \"write_fixed_latency_work_ns\": "
        << link.write_fixed_latency_work_ns << "\n"
        << indent << "    },\n"
        << indent << "    \"layer_streaming_controller\": {\n"
        << indent << "      \"present\": "
        << (streaming_controller.present ? "true" : "false") << ",\n"
        << indent << "      \"backing_admission_wait_work_ns\": "
        << streaming_controller.backing_admission_wait_work_ns << ",\n"
        << indent << "      \"backing_admission_max_wait_ns\": "
        << streaming_controller.backing_admission_max_wait_ns << ",\n"
        << indent << "      \"user_wait_work_ns\": "
        << streaming_controller.user_wait_work_ns << ",\n"
        << indent << "      \"user_max_wait_ns\": "
        << streaming_controller.user_max_wait_ns << ",\n"
        << indent << "      \"exposed_prefetch_ns\": "
        << streaming_controller.exposed_prefetch_ns << ",\n"
        << indent << "      \"hidden_prefetch_ns\": "
        << streaming_controller.hidden_prefetch_ns << ",\n"
        << indent << "      \"buffer_reuse_wait_work_ns\": "
        << streaming_controller.buffer_reuse_wait_work_ns << "\n"
        << indent << "    },\n"
        << indent << "    \"cooperative_write_controller\": {\n"
        << indent << "      \"present\": "
        << (cooperative_controller.present ? "true" : "false") << ",\n"
        << indent << "      \"full_waited_ops\": "
        << cooperative_controller.full_waited_ops << ",\n"
        << indent << "      \"full_wait_work_ns\": "
        << cooperative_controller.full_wait_work_ns << "\n"
        << indent << "    }\n"
        << indent << "  },\n"
        << indent << "  \"resource_busy\": {\n";
    const std::array<std::pair<const char*, const ResourceBusyMetric*>, 22> resources{{
        {"hbm_data_bus", &time.resource_busy.hbm_data_bus},
        {"external_controller", &time.resource_busy.external_controller},
        {"external_media", &time.resource_busy.external_media},
        {"external_link_m2s", &time.resource_busy.external_link_m2s},
        {"external_link_s2m", &time.resource_busy.external_link_s2m},
        {"hbf_logic_ingress", &time.resource_busy.hbf_logic_ingress},
        {"hbf_mapping_dram_issue",
         &time.resource_busy.hbf_mapping_dram_issue},
        {"hbf_plane_media", &time.resource_busy.hbf_plane_media},
        {"hbf_media_lane", &time.resource_busy.hbf_media_lane},
        {"hbf_subarray", &time.resource_busy.hbf_subarray},
        {"hbf_page_buffer_bank", &time.resource_busy.hbf_page_buffer_bank},
        {"hbf_flash_source_queue", &time.resource_busy.hbf_flash_source_queue},
        {"hbf_channel_command", &time.resource_busy.hbf_channel_command},
        {"hbf_channel_data", &time.resource_busy.hbf_channel_data},
        {"hbf_tsv", &time.resource_busy.hbf_tsv},
        {"hbf_sram", &time.resource_busy.hbf_sram},
        {"hbf_hbio_command", &time.resource_busy.hbf_hbio_command},
        {"hbf_hbio_data", &time.resource_busy.hbf_hbio_data},
        {"hbf_sequencer", &time.resource_busy.hbf_sequencer},
        {"hbf_ecc_issue", &time.resource_busy.hbf_ecc_issue},
        {"base_die_link_read", &time.resource_busy.base_die_link_read},
        {"base_die_link_write", &time.resource_busy.base_die_link_write},
    }};
    for (std::size_t i = 0; i < resources.size(); ++i) {
        out << indent << "    \"" << resources[i].first << "\": ";
        write_resource_busy_json(out, *resources[i].second);
        out << (i + 1 == resources.size() ? "\n" : ",\n");
    }
    out << indent << "  }\n";
    out << indent << '}';
}

void write_summary_json(
    const std::string& output_path,
    const Options& options,
    const RunProvenance& provenance,
    std::size_t op_count,
    std::uint64_t trace_footprint_bytes,
    const std::vector<ScenarioResult>& results,
    bool sanity_ok) {
    ensure_parent_dir(output_path);
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot open summary JSON output: " + output_path);
    }
    const auto resolved_hbm = make_hbm_config(options);
    const auto resolved_hbf = make_hbf_config(options);
    const auto resolved_external_backing = make_external_backing_config(options);
    // Summary provenance must round-trip every configured/modelled double;
    // six fixed decimals can silently turn small initiation intervals into
    // zero or change an ECC bandwidth sweep point.
    out << std::defaultfloat
        << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\n";
    out << "  \"schema\": {\"name\": \"hbfsim.scenario_compare.summary\", "
        << "\"version\": 16},\n";
    out << "  \"simulator\": {\n";
    out << "    \"name\": \"HBFSim\",\n";
    out << "    \"version\": \"" << json_escape(HBFSIM_VERSION) << "\",\n";
    out << "    \"git_commit\": \"" << json_escape(HBFSIM_GIT_COMMIT) << "\",\n";
    out << "    \"git_dirty\": " << (HBFSIM_GIT_DIRTY ? "true" : "false") << ",\n";
    out << "    \"git_state_captured_at\": \"cmake-configure-time\"\n";
    out << "  },\n";
    out << "  \"build\": {\n";
    out << "    \"type\": \"" << json_escape(HBFSIM_BUILD_TYPE) << "\",\n";
    out << "    \"compiler_id\": \"" << json_escape(HBFSIM_COMPILER_ID) << "\",\n";
    out << "    \"compiler_version\": \""
        << json_escape(HBFSIM_COMPILER_VERSION) << "\",\n";
    out << "    \"executable_path\": "
        << (provenance.executable_path.empty() ? "null" :
            "\"" + json_escape(provenance.executable_path) + "\"") << ",\n";
    out << "    \"executable_file_bytes\": "
        << provenance.executable_file_bytes << ",\n";
    out << "    \"executable_digest\": "
        << (provenance.executable_sha256.empty() ? "null" :
            "{\"algorithm\": \"sha256\", \"value\": \"" +
                provenance.executable_sha256 + "\"}") << "\n";
    out << "  },\n";
    out << "  \"invocation\": {\n";
    out << "    \"working_directory\": \""
        << json_escape(provenance.working_directory) << "\",\n";
    out << "    \"argv\": [";
    for (std::size_t i = 0; i < provenance.command.size(); ++i) {
        if (i != 0) out << ", ";
        out << "\"" << json_escape(provenance.command[i]) << "\"";
    }
    out << "]\n";
    out << "  },\n";
    out << "  \"workload\": {\n";
    out << "    \"source\": \""
        << (options.synthetic_sequential_read_bytes ?
            "synthetic-sequential-read-v1" : "trace-file")
        << "\",\n";
    out << "    \"trace_path\": "
        << (options.trace_path ? "\"" + json_escape(*options.trace_path) + "\"" : "null")
        << ",\n";
    out << "    \"trace_file_bytes\": " << provenance.trace_file_bytes << ",\n";
    out << "    \"trace_digest\": {\"algorithm\": \"sha256\", \"value\": \""
        << provenance.trace_sha256 << "\"},\n";
    out << "    \"initial_image_trace_path\": "
        << (options.initial_image_trace_path ?
            "\"" + json_escape(*options.initial_image_trace_path) + "\"" :
            "null")
        << ",\n";
    out << "    \"initial_image_trace_file_bytes\": "
        << provenance.initial_image_trace_file_bytes << ",\n";
    out << "    \"initial_image_trace_digest\": ";
    if (provenance.initial_image_trace_sha256.empty()) {
        out << "null\n";
    } else {
        out << "{\"algorithm\": \"sha256\", \"value\": \""
            << provenance.initial_image_trace_sha256 << "\"}\n";
    }
    out << "  },\n";
    out << "  \"validation\": {\n";
    out << "    \"status\": \"exploratory_unattached\",\n";
    out << "    \"certificate\": null\n";
    out << "  },\n";
    out << "  \"sanity\": \"" << (sanity_ok ? "PASS" : "FAIL") << "\",\n";
    out << "  \"config\": {\n";
    out << "    \"ops\": " << op_count << ",\n";
    out << "    \"trace\": " << (options.trace_path ? "\"" + json_escape(*options.trace_path) + "\"" : "null") << ",\n";
    out << "    \"expected_trace_sha256\": "
        << (options.expected_trace_sha256 ?
            "\"" + *options.expected_trace_sha256 + "\"" :
            "null")
        << ",\n";
    out << "    \"expected_trace_bytes\": ";
    if (options.expected_trace_file_bytes) {
        out << *options.expected_trace_file_bytes;
    } else {
        out << "null";
    }
    out << ",\n";
    out << "    \"initial_image_trace\": "
        << (options.initial_image_trace_path ?
            "\"" + json_escape(*options.initial_image_trace_path) + "\"" :
            "null")
        << ",\n";
    out << "    \"synthetic_sequential_read_bytes\": ";
    if (options.synthetic_sequential_read_bytes) {
        out << *options.synthetic_sequential_read_bytes;
    } else {
        out << "null";
    }
    out << ",\n";
    out << "    \"synthetic_sequential_read_base\": "
        << options.synthetic_sequential_read_base << ",\n";
    out << "    \"generate_llm\": "
        << (options.generate_llm_path ? "\"" + json_escape(*options.generate_llm_path) + "\"" : "null")
        << ",\n";
    out << "    \"generate_semantic_llm\": "
        << (options.generate_semantic_llm_path ?
            "\"" + json_escape(*options.generate_semantic_llm_path) + "\"" : "null")
        << ",\n";
    out << "    \"line_size\": " << options.line_size << ",\n";
    out << "    \"trace_unique_line_bytes\": " << trace_footprint_bytes << ",\n";
    out << "    \"interarrival_ns\": " << options.interarrival_ns << ",\n";
    out << "    \"max_ops\": " << options.max_ops << ",\n";
    const auto resolved_trace_mode = effective_trace_mode(options);
    out << "    \"trace_mode\": \""
        << (resolved_trace_mode == TraceMode::Off ? "off" :
            resolved_trace_mode == TraceMode::Summary ? "summary" :
            resolved_trace_mode == TraceMode::Sampled ? "sampled" : "full")
        << "\",\n";
    out << "    \"requested_scenarios\": [";
    for (std::size_t i = 0; i < options.scenarios.size(); ++i) {
        if (i != 0) out << ", ";
        out << "\"" << json_escape(options.scenarios[i]) << "\"";
    }
    out << "],\n";
    out << "    \"hbm_capacity_bytes\": " << options.hbm_capacity_bytes << ",\n";
    out << "    \"flat_hbm_bytes\": " << options.flat_hbm_bytes << ",\n";
    out << "    \"static_direct_hbm_bytes\": "
        << options.static_direct_hbm_bytes << ",\n";
    out << "    \"hbf_capacity_bytes\": " << hbf_capacity_bytes(options) << ",\n";
    out << "    \"hbf_to_hbm_capacity_ratio\": "
        << (static_cast<long double>(hbf_capacity_bytes(options)) /
            static_cast<long double>(options.hbm_capacity_bytes)) << ",\n";
    out << "    \"layer_buffer_bytes\": "
        << options.layer_buffer_bytes << ",\n";
    out << "    \"explicit_residency_contract\": "
        << (options.explicit_residency_contract ? "true" : "false")
        << ",\n";
    out << "    \"residency_contract\": ";
    if (options.explicit_residency_contract) {
        out << "{\n";
        out << "      \"page_size_bytes\": "
            << *options.residency_page_size_bytes << ",\n";
        out << "      \"unique_resident_footprint_bytes\": "
            << *options.residency_unique_footprint_bytes << ",\n";
        out << "      \"immutable_weight_bytes\": "
            << *options.residency_immutable_weight_bytes << ",\n";
        out << "      \"immutable_weight_pages\": "
            << *options.residency_immutable_weight_pages << ",\n";
        out << "      \"static_weight_resident_pages\": "
            << *options.residency_static_weight_pages << ",\n";
        out << "      \"runtime_overhead_bytes\": "
            << *options.residency_runtime_overhead_bytes << ",\n";
        out << "      \"block_table_bytes\": "
            << *options.residency_block_table_bytes << ",\n";
        out << "      \"active_buffer_bytes_per_slot\": "
            << *options.residency_active_buffer_bytes << ",\n";
        out << "      \"kv_region_begin\": "
            << *options.residency_kv_region_begin << ",\n";
        out << "      \"kv_block_stride_bytes\": "
            << *options.residency_kv_block_stride_bytes << ",\n";
        out << "      \"logical_kv_blocks\": "
            << *options.residency_logical_kv_blocks << ",\n";
        out << "      \"hot_kv_blocks\": "
            << *options.residency_hot_kv_blocks << "\n";
        out << "    },\n";
    } else {
        out << "null,\n";
    }
    out << "    \"behavioral_hbm_bytes\": "
        << (options.behavioral_hbm_bytes == 0 ?
            options.hbm_capacity_bytes :
            options.behavioral_hbm_bytes)
        << ",\n";
    out << "    \"behavioral_promotion_threshold\": "
        << options.behavioral_promotion_threshold << ",\n";
    out << "    \"behavioral_history_pages\": "
        << options.behavioral_history_pages << ",\n";
    out << "    \"max_outstanding_requests\": "
        << options.max_outstanding_requests << ",\n";
    out << "    \"max_hbm_outstanding_requests\": "
        << options.max_hbm_outstanding_requests << ",\n";
    out << "    \"max_hbf_outstanding_requests\": "
        << options.max_hbf_outstanding_requests << ",\n";
    out << "    \"address_heatmap_bins\": "
        << options.address_heatmap_bins << ",\n";
    out << "    \"hbf_hbm_write_buffer_bytes\": "
        << options.hbf_hbm_write_buffer_bytes << ",\n";
    out << "    \"hbm_cooperative_region_base_addr\": "
        << hbm_cooperative_region_base_addr(options) << ",\n";
    out << "    \"hbf_hbm_write_buffer_destage\": \""
        << json_escape(options.hbf_hbm_write_buffer_destage) << "\",\n";
    out << "    \"base_die_link\": {\n";
    out << "      \"read_bw_GBps\": " << options.base_die_link_read_bw_GBps << ",\n";
    out << "      \"write_bw_GBps\": " << options.base_die_link_write_bw_GBps << ",\n";
    out << "      \"latency_ns\": " << options.base_die_link_latency_ns << "\n";
    out << "    },\n";
    out << "    \"external_backing\": {\n";
    out << "      \"kind\": \""
        << hbfsim::physical::external::to_string(resolved_external_backing.kind)
        << "\",\n";
    out << "      \"capacity_bytes\": "
        << resolved_external_backing.capacity_bytes << ",\n";
    out << "      \"page_size_bytes\": "
        << resolved_external_backing.page_size_bytes << ",\n";
    out << "      \"media_channels\": "
        << resolved_external_backing.media_channels << ",\n";
    out << "      \"max_outstanding_requests\": "
        << resolved_external_backing.max_outstanding_requests << ",\n";
    out << "      \"controller_issue_ns\": "
        << resolved_external_backing.controller_issue_ns << ",\n";
    out << "      \"controller_processing_ns\": "
        << resolved_external_backing.controller_processing_ns << ",\n";
    out << "      \"media_read_latency_ns\": "
        << resolved_external_backing.media_read_latency_ns << ",\n";
    out << "      \"media_write_latency_ns\": "
        << resolved_external_backing.media_write_latency_ns << ",\n";
    out << "      \"media_read_bandwidth_GBps\": "
        << resolved_external_backing.media_read_bandwidth_GBps << ",\n";
    out << "      \"media_write_bandwidth_GBps\": "
        << resolved_external_backing.media_write_bandwidth_GBps << ",\n";
    out << "      \"m2s_bandwidth_GBps\": "
        << resolved_external_backing.m2s_bandwidth_GBps << ",\n";
    out << "      \"s2m_bandwidth_GBps\": "
        << resolved_external_backing.s2m_bandwidth_GBps << ",\n";
    out << "      \"one_way_propagation_ns\": "
        << resolved_external_backing.one_way_propagation_ns << ",\n";
    out << "      \"command_bytes\": "
        << resolved_external_backing.command_bytes << ",\n";
    out << "      \"completion_bytes\": "
        << resolved_external_backing.completion_bytes << "\n";
    out << "    },\n";
    out << "    \"llm\": ";
    if (options.generate_llm_path || options.generate_semantic_llm_path) {
        out << "{\n";
        out << "      \"tokens\": " << options.llm_tokens << ",\n";
        out << "      \"layers\": " << options.llm_layers << ",\n";
        out << "      \"weight_base\": " << options.llm_weight_base << ",\n";
        out << "      \"kv_base\": " << options.llm_kv_base << ",\n";
        out << "      \"scratch_base\": " << options.llm_scratch_base << "\n";
        out << "    },\n";
    } else {
        out << "null,\n";
    }
    out << "    \"hbm\": {\n";
    out << "      \"capacity_bytes\": " << options.hbm_capacity_bytes << ",\n";
    out << "      \"stacks\": " << options.hbm_stacks << ",\n";
    out << "      \"channels\": " << options.hbm_channels << ",\n";
    out << "      \"pseudo_channels\": " << options.hbm_pseudo_channels << ",\n";
    out << "      \"address_mapping_scheme\": \""
        << resolved_hbm.address_mapping_scheme() << "\",\n";
    out << "      \"bank_groups_per_pseudo_channel\": "
        << options.hbm_bank_groups_per_pseudo_channel << ",\n";
    out << "      \"banks_per_pseudo_channel\": "
        << static_cast<std::uint64_t>(options.hbm_bank_groups_per_pseudo_channel) *
            options.hbm_banks_per_group << ",\n";
    out << "      \"banks_per_group\": " << options.hbm_banks_per_group << ",\n";
    out << "      \"channel_row_size_bytes\": "
        << resolved_hbm.channel_row_size_bytes << ",\n";
    out << "      \"row_size_bytes_per_pseudo_channel\": "
        << resolved_hbm.row_size_bytes() << ",\n";
    out << "      \"channel_width_bits\": "
        << resolved_hbm.channel_width_bits << ",\n";
    out << "      \"pseudo_channel_width_bits\": "
        << resolved_hbm.pseudo_channel_width_bits() << ",\n";
    out << "      \"pin_rate_Gbps\": " << resolved_hbm.pin_rate_Gbps << ",\n";
    out << "      \"data_rate_per_command_clock\": "
        << resolved_hbm.data_rate_per_command_clock << ",\n";
    out << "      \"command_clock_MHz\": "
        << resolved_hbm.command_clock_MHz() << ",\n";
    out << "      \"command_clock_period_ns\": "
        << resolved_hbm.command_clock_period_ns() << ",\n";
    out << "      \"burst_length\": " << resolved_hbm.burst_length << ",\n";
    out << "      \"burst_bytes\": " << resolved_hbm.burst_bytes() << ",\n";
    out << "      \"burst_duration_ns\": "
        << resolved_hbm.burst_duration_ns() << ",\n";
    out << "      \"pseudo_channel_bw_GBps\": "
        << resolved_hbm.pseudo_channel_bandwidth_GBps() << ",\n";
    out << "      \"channel_bw_GBps\": "
        << resolved_hbm.channel_bandwidth_GBps() << ",\n";
    out << "      \"stack_peak_bw_GBps\": "
        << resolved_hbm.channel_bandwidth_GBps() * resolved_hbm.channels_per_stack
        << ",\n";
    out << "      \"system_peak_bw_GBps\": "
        << resolved_hbm.channel_bandwidth_GBps() * resolved_hbm.channels_per_stack *
            resolved_hbm.stacks << ",\n";
    out << "      \"address_mapping_ns\": "
        << resolved_hbm.address_mapping_ns << ",\n";
    out << "      \"timing_minima_issue_semantics\": "
        << "\"absolute-ns minima rounded up at command-clock issue\",\n";
    out << "      \"trcdrd_ns\": " << resolved_hbm.tRCDRD_ns << ",\n";
    out << "      \"trcdwr_ns\": " << resolved_hbm.tRCDWR_ns << ",\n";
    out << "      \"tcl_ns\": " << resolved_hbm.tCL_ns << ",\n";
    out << "      \"tcwl_ns\": " << resolved_hbm.tCWL_ns << ",\n";
    out << "      \"trp_ns\": " << resolved_hbm.tRP_ns << ",\n";
    out << "      \"tras_ns\": " << resolved_hbm.tRAS_ns << ",\n";
    out << "      \"trc_ns\": " << resolved_hbm.tRC_ns << ",\n";
    out << "      \"twr_ns\": " << resolved_hbm.tWR_ns << ",\n";
    out << "      \"trtp_ns\": " << resolved_hbm.tRTP_ns << ",\n";
    out << "      \"tccd_s_cycles\": " << resolved_hbm.tCCD_S_cycles << ",\n";
    out << "      \"tccd_l_cycles\": " << resolved_hbm.tCCD_L_cycles << ",\n";
    out << "      \"tccd_s_ns\": " << resolved_hbm.tCCD_S_ns() << ",\n";
    out << "      \"tccd_l_ns\": " << resolved_hbm.tCCD_L_ns() << ",\n";
    out << "      \"trrd_s_ns\": " << resolved_hbm.tRRD_S_ns << ",\n";
    out << "      \"trrd_l_ns\": " << resolved_hbm.tRRD_L_ns << ",\n";
    out << "      \"tfaw_ns\": " << resolved_hbm.tFAW_ns << ",\n";
    out << "      \"twtr_s_ns\": " << resolved_hbm.tWTR_S_ns << ",\n";
    out << "      \"twtr_l_ns\": " << resolved_hbm.tWTR_L_ns << ",\n";
    out << "      \"trtw_ns\": " << resolved_hbm.tRTW_ns << ",\n";
    out << "      \"effective_trcdrd_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tRCDRD_ns) << ",\n";
    out << "      \"effective_trcdwr_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tRCDWR_ns) << ",\n";
    out << "      \"effective_tcl_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tCL_ns) << ",\n";
    out << "      \"effective_tcwl_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tCWL_ns) << ",\n";
    out << "      \"effective_trp_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tRP_ns) << ",\n";
    const double effective_tras_ns =
        resolved_hbm.command_aligned_time_ns(resolved_hbm.tRAS_ns);
    const double effective_trp_ns =
        resolved_hbm.command_aligned_time_ns(resolved_hbm.tRP_ns);
    out << "      \"effective_tras_ns\": " << effective_tras_ns << ",\n";
    out << "      \"effective_trc_ns\": "
        << std::max(
               resolved_hbm.command_aligned_time_ns(resolved_hbm.tRC_ns),
               effective_tras_ns + effective_trp_ns)
        << ",\n";
    out << "      \"effective_twr_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tWR_ns) << ",\n";
    out << "      \"effective_trtp_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tRTP_ns) << ",\n";
    out << "      \"effective_trrd_s_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tRRD_S_ns) << ",\n";
    out << "      \"effective_trrd_l_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tRRD_L_ns) << ",\n";
    out << "      \"effective_tfaw_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tFAW_ns) << ",\n";
    out << "      \"effective_twtr_s_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tWTR_S_ns) << ",\n";
    out << "      \"effective_twtr_l_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tWTR_L_ns) << ",\n";
    out << "      \"effective_trtw_ns\": "
        << resolved_hbm.command_aligned_time_ns(resolved_hbm.tRTW_ns) << ",\n";
    out << "      \"refresh_enabled\": "
        << (options.hbm_refresh_enabled ? "true" : "false") << ",\n";
    out << "      \"same_bank_refresh\": "
        << (options.hbm_same_bank_refresh ? "true" : "false") << ",\n";
    out << "      \"trefi_ns\": " << options.hbm_trefi_ns << ",\n";
    out << "      \"trfc_ns\": " << options.hbm_trfc_ns << ",\n";
    out << "      \"trfcsb_ns\": " << options.hbm_trfcsb_ns << ",\n";
    out << "      \"trrefd_ns\": " << resolved_hbm.tRREFD_ns << ",\n";
    out << "      \"queue_depth\": " << options.hbm_queue_depth << ",\n";
    out << "      \"frfcfs_cap_ns\": " << options.hbm_frfcfs_cap_ns << "\n";
    out << "    },\n";
    out << "    \"hbf\": {\n";
    out << "      \"capacity_bytes\": " << hbf_capacity_bytes(options) << ",\n";
    out << "      \"placement_mapping_scheme\": \""
        << hbfsim::physical::hbf::kPlacementMappingScheme << "\",\n";
    out << "      \"stacks\": " << options.hbf_stacks << ",\n";
    out << "      \"channels\": " << options.hbf_channels << ",\n";
    out << "      \"dies_per_channel\": " << options.hbf_dies_per_channel << ",\n";
    out << "      \"planes_per_die\": " << options.hbf_planes_per_die << ",\n";
    out << "      \"blocks_per_plane\": " << options.hbf_blocks_per_plane << ",\n";
    out << "      \"pages_per_block\": " << options.hbf_pages_per_block << ",\n";
    out << "      \"page_size\": " << options.hbf_page_size << ",\n";
    out << "      \"oob_bytes\": " << options.hbf_oob_bytes << ",\n";
    out << "      \"media_lanes_per_plane\": " << options.hbf_media_lanes_per_plane << ",\n";
    out << "      \"subarrays_per_plane\": " << options.hbf_subarrays_per_plane << ",\n";
    out << "      \"page_buffer_banks_per_plane\": "
        << options.hbf_page_buffer_banks_per_plane << ",\n";
    out << "      \"read_ns\": " << options.hbf_read_ns << ",\n";
    out << "      \"program_ns\": " << options.hbf_program_ns << ",\n";
    out << "      \"erase_ns\": " << options.hbf_erase_ns << ",\n";
    out << "      \"program_verify_ns\": "
        << resolved_hbf.t_program_verify_ns << ",\n";
    out << "      \"ecc_decode_latency_ns\": "
        << resolved_hbf.ecc_decode_latency_ns << ",\n";
    out << "      \"ecc_encode_latency_ns\": "
        << resolved_hbf.ecc_encode_latency_ns << ",\n";
    out << "      \"ecc_decode_raw_bw_GBps_per_die\": "
        << resolved_hbf.ecc_decode_raw_bandwidth_GBps_per_die << ",\n";
    out << "      \"ecc_encode_raw_bw_GBps_per_die\": "
        << resolved_hbf.ecc_encode_raw_bandwidth_GBps_per_die << ",\n";
    out << "      \"ecc_codeword_size_bytes\": "
        << resolved_hbf.page_size_bytes + resolved_hbf.oob_bytes_per_page << ",\n";
    out << "      \"ecc_decode_initiation_ns\": "
        << hbfsim::physical::transfer_time_ns(
            resolved_hbf.page_size_bytes + resolved_hbf.oob_bytes_per_page,
            resolved_hbf.ecc_decode_raw_bandwidth_GBps_per_die) << ",\n";
    out << "      \"ecc_encode_initiation_ns\": "
        << hbfsim::physical::transfer_time_ns(
            resolved_hbf.page_size_bytes + resolved_hbf.oob_bytes_per_page,
            resolved_hbf.ecc_encode_raw_bandwidth_GBps_per_die) << ",\n";
    out << "      \"ecc_issue_topology\": \"shared-per-die\",\n";
    out << "      \"bandwidth_semantics\": {"
        << "\"channel\":\"raw-codeword-per-channel\","
        << "\"tsv\":\"shared-command-and-raw-codeword-per-stack\","
        << "\"hbio\":\"decoded-payload-per-stack\","
        << "\"media_lane\":\"raw-codeword-per-lane\","
        << "\"page_buffer\":\"raw-codeword-per-bank\","
        << "\"logic_sram\":\"decoded-payload-per-stack\"},\n";
    out << "      \"channel_bw_GBps\": " << options.hbf_channel_bw_GBps << ",\n";
    out << "      \"hbio_bw_GBps\": " << options.hbf_hbio_bw_GBps << ",\n";
    out << "      \"tsv_bw_GBps\": " << options.hbf_tsv_bw_GBps << ",\n";
    out << "      \"media_lane_bw_GBps\": " << options.hbf_media_lane_bw_GBps << ",\n";
    out << "      \"logic_sram_bw_GBps\": "
        << resolved_hbf.logic_sram_bandwidth_GBps << ",\n";
    out << "      \"page_buffer_bw_GBps\": " << options.hbf_page_buffer_bw_GBps << ",\n";
    out << "      \"command_address_bytes\": "
        << resolved_hbf.command_address_bytes << ",\n";
    out << "      \"ctrl_dram_bytes\": " << resolved_hbf.ctrl_dram_bytes << ",\n";
    out << "      \"ctrl_dram_latency_ns\": "
        << options.hbf_ctrl_dram_latency_ns << ",\n";
    out << "      \"ctrl_dram_issue_ns\": "
        << options.hbf_ctrl_dram_issue_ns << ",\n";
    out << "      \"flash_tsu_issue_ns\": "
        << resolved_hbf.flash_tsu_issue_ns << ",\n";
    out << "      \"logic_scheduler_issue_ns\": "
        << options.hbf_logic_scheduler_issue_ns << ",\n";
    out << "      \"address_generation_ns\": "
        << resolved_hbf.address_generation_ns << ",\n";
    out << "      \"mapping_update_ns\": "
        << resolved_hbf.mapping_update_ns << ",\n";
    out << "      \"free_page_allocation_ns\": "
        << resolved_hbf.free_page_allocation_ns << ",\n";
    out << "      \"mapping_entries_per_page\": "
        << resolved_hbf.mapping_entries_per_page << ",\n";
    out << "      \"batch_activation\": "
        << (options.hbf_batch_activation ? "true" : "false") << ",\n";
    out << "      \"read_buffer_pages\": " << options.hbf_read_buffer_pages << ",\n";
    out << "      \"page_read_queue_depth_per_stack\": "
        << options.hbf_page_read_queue_depth_per_stack << ",\n";
    out << "      \"auto_gc_enabled\": "
        << (resolved_hbf.auto_gc_enabled ? "true" : "false") << ",\n";
    out << "      \"gc_low_watermark_pages\": " << options.hbf_gc_low_watermark_pages << ",\n";
    out << "      \"gc_hard_watermark_pages\": " << options.hbf_gc_hard_watermark_pages << ",\n";
    out << "      \"gc_reserved_free_blocks_per_plane\": "
        << options.hbf_gc_reserved_free_blocks_per_plane << ",\n";
    out << "      \"gc_wear_leveling_weight\": " << options.hbf_gc_wear_leveling_weight << ",\n";
    out << "      \"write_coalescing\": " << (options.hbf_write_coalescing ? "true" : "false") << ",\n";
    out << "      \"write_buffer_completion_requires_flush\": "
        << (options.hbf_write_buffer_completion_requires_flush ? "true" : "false") << ",\n";
    out << "      \"write_buffer_pages\": " << options.hbf_write_buffer_pages << ",\n";
    out << "      \"write_buffer_flush_threshold_pages\": "
        << options.hbf_write_buffer_flush_threshold_pages << "\n";
    out << "    }\n";
    out << "  },\n";
    out << "  \"scenarios\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        out << "    {\n";
        out << "      \"name\": \"" << json_escape(result.name) << "\",\n";
        out << "      \"ops\": " << result.ops << ",\n";
        out << "      \"reads\": " << result.reads << ",\n";
        out << "      \"writes\": " << result.writes << ",\n";
        out << "      \"logical_bytes\": " << result.logical_bytes << ",\n";
        out << "      \"hbm_accesses\": " << result.hbm_accesses << ",\n";
        out << "      \"hbf_accesses\": " << result.hbf_accesses << ",\n";
        out << "      \"external_accesses\": "
            << result.external_accesses << ",\n";
        out << "      \"hbm_user_accesses\": " << result.hbm_user_accesses << ",\n";
        out << "      \"hbf_user_accesses\": " << result.hbf_user_accesses << ",\n";
        out << "      \"hbf_direct_user_ops\": " << result.hbf_direct_user_ops << ",\n";
        out << "      \"hbm_background_accesses\": " << result.hbm_background_accesses << ",\n";
        out << "      \"hbf_background_accesses\": " << result.hbf_background_accesses << ",\n";
        out << "      \"external_background_accesses\": "
            << result.external_background_accesses << ",\n";
        out << "      \"behavioral_tiering\": ";
        if (result.behavioral_tiering_stats) {
            const auto& s = *result.behavioral_tiering_stats;
            out << "{\n";
            out << "        \"policy\": \""
                << hbfsim::physical::hybrid::to_string(
                       s.admission_policy)
                << "\",\n";
            out << "        \"semantic_inputs_consumed\": "
                << (s.semantic_inputs_consumed ? "true" : "false")
                << ",\n";
            out << "        \"hbm_tier_pages\": "
                << s.hbm_tier_pages << ",\n";
            out << "        \"hbm_tier_bytes\": "
                << s.hbm_tier_bytes << ",\n";
            out << "        \"promotion_threshold\": "
                << s.promotion_threshold << ",\n";
            out << "        \"history_capacity_pages\": "
                << s.history_capacity_pages << ",\n";
            out << "        \"page_observations\": "
                << s.page_observations << ",\n";
            out << "        \"hbm_hits\": " << s.hbm_hits << ",\n";
            out << "        \"hbf_bypasses\": "
                << s.hbf_bypasses << ",\n";
            out << "        \"cold_misses\": "
                << s.cold_misses << ",\n";
            out << "        \"history_hits\": "
                << s.history_hits << ",\n";
            out << "        \"promotions\": "
                << s.promotions << ",\n";
            out << "        \"promotions_with_backing_fill\": "
                << s.promotions_with_backing_fill << ",\n";
            out << "        \"promotions_without_backing_fill\": "
                << s.promotions_without_backing_fill << ",\n";
            out << "        \"clean_evictions\": "
                << s.clean_evictions << ",\n";
            out << "        \"dirty_evictions\": "
                << s.dirty_evictions << ",\n";
            out << "        \"history_evictions\": "
                << s.history_evictions << ",\n";
            out << "        \"hbm_foreground_bytes\": "
                << s.hbm_foreground_bytes << ",\n";
            out << "        \"hbf_bypass_bytes\": "
                << s.hbf_bypass_bytes << ",\n";
            out << "        \"backing_fill_pages\": "
                << s.backing_fill_pages << ",\n";
            out << "        \"backing_fill_bytes\": "
                << s.backing_fill_bytes << ",\n";
            out << "        \"hbm_install_pages\": "
                << s.hbm_install_pages << ",\n";
            out << "        \"hbm_install_bytes\": "
                << s.hbm_install_bytes << ",\n";
            out << "        \"dirty_writeback_pages\": "
                << s.dirty_writeback_pages << ",\n";
            out << "        \"dirty_writeback_bytes\": "
                << s.dirty_writeback_bytes << ",\n";
            out << "        \"drain_writeback_pages\": "
                << s.drain_writeback_pages << ",\n";
            out << "        \"drain_writeback_bytes\": "
                << s.drain_writeback_bytes << ",\n";
            out << "        \"peak_resident_pages\": "
                << s.peak_resident_pages << ",\n";
            out << "        \"final_resident_pages\": "
                << s.final_resident_pages << ",\n";
            out << "        \"final_dirty_pages\": "
                << s.final_dirty_pages << ",\n";
            out << "        \"peak_history_pages\": "
                << s.peak_history_pages << ",\n";
            out << "        \"peak_installing_pages\": "
                << s.peak_installing_pages << ",\n";
            out << "        \"capacity_stalled_promotions\": "
                << s.capacity_stalled_promotions << ",\n";
            out << "        \"capacity_stall_work_ns\": "
                << s.capacity_stall_work_ns << ",\n";
            out << "        \"transition_waited_transactions\": "
                << s.transition_waited_transactions << ",\n";
            out << "        \"transition_wait_work_ns\": "
                << s.transition_wait_work_ns << ",\n";
            out << "        \"transition_max_wait_ns\": "
                << s.transition_max_wait_ns << ",\n";
            out << "        \"decision_fingerprint\": "
                << s.decision_fingerprint << "\n";
            out << "      },\n";
        } else {
            out << "null,\n";
        }
        out << "      \"layer_streaming\": ";
        if (result.layer_streaming_stats) {
            const auto& s = *result.layer_streaming_stats;
            out << "{\n";
            out << "        \"mode\": \""
                << layer_streaming_mode_name(s) << "\",\n";
            out << "        \"backing\": \""
                << hbfsim::physical::hybrid::to_string(s.backing)
                << "\",\n";
            out << "        \"residency_policy\": \""
                << layer_streaming_residency_policy_name(s)
                << "\",\n";
            out << "        \"compact_resident_mapping\": "
                << (s.compact_resident_mapping ? "true" : "false")
                << ",\n";
            out << "        \"semantic_inputs_consumed\": "
                << (s.semantic_inputs_consumed ? "true" : "false")
                << ",\n";
            out << "        \"explicit_residency_contract\": "
                << (s.explicit_residency_contract ? "true" : "false")
                << ",\n";
            out << "        \"address_footprint_bytes\": "
                << s.address_footprint_bytes << ",\n";
            out << "        \"unique_resident_footprint_pages\": "
                << s.unique_resident_footprint_pages << ",\n";
            out << "        \"unique_resident_footprint_bytes\": "
                << s.unique_resident_footprint_bytes << ",\n";
            out << "        \"capacity_pressure_basis_bytes\": "
                << s.capacity_pressure_basis_bytes << ",\n";
            out << "        \"footprint_page_rounding_bytes\": "
                << s.footprint_page_rounding_bytes << ",\n";
            out << "        \"hbm_capacity_bytes\": "
                << s.hbm_capacity_bytes << ",\n";
            out << "        \"hbm_capacity_pressure\": "
                << s.hbm_capacity_pressure << ",\n";
            out << "        \"layers\": " << s.layers << ",\n";
            out << "        \"explicit_layer_requests\": "
                << s.explicit_layer_requests << ",\n";
            out << "        \"explicit_compute_layers\": "
                << s.explicit_compute_layers << ",\n";
            out << "        \"compute_work_ns\": "
                << s.compute_work_ns << ",\n";
            out << "        \"hbm_only_resident_pages\": "
                << s.hbm_only_resident_pages << ",\n";
            out << "        \"hbm_only_resident_bytes\": "
                << s.hbm_only_resident_bytes << ",\n";
            out << "        \"hot_kv_candidate_pages\": "
                << s.hot_kv_candidate_pages << ",\n";
            out << "        \"hot_kv_candidate_bytes\": "
                << s.hot_kv_candidate_bytes << ",\n";
            out << "        \"hot_kv_resident_pages\": "
                << s.hot_kv_resident_pages << ",\n";
            out << "        \"hot_kv_resident_bytes\": "
                << s.hot_kv_resident_bytes << ",\n";
            out << "        \"data_pages\": " << s.data_pages << ",\n";
            out << "        \"data_bytes\": " << s.data_bytes << ",\n";
            out << "        \"model_weight_resident_pages\": "
                << s.model_weight_resident_pages << ",\n";
            out << "        \"model_weight_resident_bytes\": "
                << s.model_weight_resident_bytes << ",\n";
            out << "        \"model_weight_backing_pages\": "
                << s.model_weight_backing_pages << ",\n";
            out << "        \"model_weight_backing_bytes\": "
                << s.model_weight_backing_bytes << ",\n";
            out << "        \"cold_kv_backing_pages\": "
                << s.cold_kv_backing_pages << ",\n";
            out << "        \"cold_kv_backing_bytes\": "
                << s.cold_kv_backing_bytes << ",\n";
            out << "        \"unknown_backing_pages\": "
                << s.unknown_backing_pages << ",\n";
            out << "        \"unknown_backing_bytes\": "
                << s.unknown_backing_bytes << ",\n";
            out << "        \"backing_unique_pages\": "
                << s.backing_unique_pages << ",\n";
            out << "        \"backing_unique_bytes\": "
                << s.backing_unique_bytes << ",\n";
            out << "        \"resident_physical_pages\": "
                << s.resident_physical_pages << ",\n";
            out << "        \"resident_physical_bytes\": "
                << s.resident_physical_bytes << ",\n";
            out << "        \"effective_layer_buffer_pages\": "
                << s.effective_layer_buffer_pages << ",\n";
            out << "        \"effective_layer_buffer_bytes\": "
                << s.effective_layer_buffer_bytes << ",\n";
            out << "        \"unused_hbm_pages\": "
                << s.unused_hbm_pages << ",\n";
            out << "        \"unused_hbm_bytes\": "
                << s.unused_hbm_bytes << ",\n";
            out << "        \"streamed_pages\": " << s.streamed_pages << ",\n";
            out << "        \"streamed_bytes\": " << s.streamed_bytes << ",\n";
            out << "        \"foreground_resident_page_accesses\": "
                << s.foreground_resident_page_accesses << ",\n";
            out << "        \"foreground_buffer_page_accesses\": "
                << s.foreground_buffer_page_accesses << ",\n";
            out << "        \"dirty_pages_written_back\": "
                << s.dirty_pages_written_back << ",\n";
            out << "        \"writeback_bytes\": " << s.writeback_bytes << ",\n";
            out << "        \"max_layer_data_pages\": "
                << s.max_layer_data_pages << ",\n";
            out << "        \"max_layer_data_bytes\": "
                << s.max_layer_data_bytes << ",\n";
            out << "        \"immutable_weight_logical_bytes\": "
                << s.immutable_weight_logical_bytes << ",\n";
            out << "        \"runtime_overhead_logical_bytes\": "
                << s.runtime_overhead_logical_bytes << ",\n";
            out << "        \"block_table_logical_bytes\": "
                << s.block_table_logical_bytes << ",\n";
            out << "        \"active_buffer_logical_bytes_per_slot\": "
                << s.active_buffer_logical_bytes_per_slot << ",\n";
            out << "        \"residency_page_size_bytes\": "
                << s.residency_page_size_bytes << ",\n";
            out << "        \"kv_block_stride_bytes\": "
                << s.kv_block_stride_bytes << ",\n";
            out << "        \"logical_kv_blocks\": "
                << s.logical_kv_blocks << ",\n";
            out << "        \"hot_kv_blocks\": "
                << s.hot_kv_blocks << ",\n";
            out << "        \"cold_kv_blocks\": "
                << s.cold_kv_blocks << ",\n";
            out << "        \"backing_request_credit_limit\": "
                << s.backing_request_credit_limit << ",\n";
            out << "        \"backing_max_inflight_requests\": "
                << s.backing_max_inflight_requests << ",\n";
            out << "        \"backing_admission_waited_requests\": "
                << s.backing_admission_waited_requests << ",\n";
            out << "        \"backing_admission_wait_work_ns\": "
                << s.backing_admission_wait_work_ns << ",\n";
            out << "        \"backing_admission_max_wait_ns\": "
                << s.backing_admission_max_wait_ns << ",\n";
            out << "        \"user_waited_ops\": " << s.user_waited_ops << ",\n";
            out << "        \"user_wait_work_ns\": " << s.user_wait_work_ns << ",\n";
            out << "        \"user_max_wait_ns\": " << s.user_max_wait_ns << ",\n";
            out << "        \"exposed_prefetch_ns\": "
                << s.exposed_prefetch_ns << ",\n";
            out << "        \"hidden_prefetch_ns\": "
                << s.hidden_prefetch_ns << ",\n";
            out << "        \"buffer_reuse_wait_work_ns\": "
                << s.buffer_reuse_wait_work_ns << "\n";
            out << "      },\n";
        } else {
            out << "null,\n";
        }
        out << "      \"hybrid_path\": {\n";
        out << "        \"background_hbf_writes\": "
            << result.background_hbf_writes << ",\n";
        out << "        \"hbf_static_read_bytes\": " << result.hbf_static_read_bytes << ",\n";
        out << "        \"hbf_logical_read_bytes\": " << result.hbf_logical_read_bytes << ",\n";
        out << "        \"hbf_backing_write_bytes\": " << result.hbf_backing_write_bytes << ",\n";
        out << "        \"external_backing_read_bytes\": "
            << result.external_backing_read_bytes << ",\n";
        out << "        \"external_backing_write_bytes\": "
            << result.external_backing_write_bytes << ",\n";
        out << "        \"hbm_foreground_bytes\": " << result.hbm_foreground_bytes << ",\n";
        out << "        \"hbm_streaming_write_bytes\": "
            << result.hbm_streaming_write_bytes << ",\n";
        out << "        \"base_die_link_read_bytes\": "
            << result.base_die_link_stats.read_bytes << ",\n";
        out << "        \"base_die_link_write_bytes\": "
            << result.base_die_link_stats.write_bytes << ",\n";
        out << "        \"hbm_write_buffer_user_write_bytes\": "
            << result.hbm_write_buffer_user_write_bytes << ",\n";
        out << "        \"hbm_write_buffer_destaged_bytes\": " << result.hbm_write_buffer_destaged_bytes << ",\n";
        out << "        \"hbm_write_buffer_peak_bytes\": " << result.hbm_write_buffer_peak_bytes << ",\n";
        out << "        \"hbm_write_buffer_full_waits\": " << result.hbm_write_buffer_full_waits << ",\n";
        out << "        \"hbm_write_buffer_full_wait_work_ns\": "
            << result.hbm_write_buffer_wait_ns << "\n";
        out << "      },\n";
        write_time_breakdown_json(out, result, "      ");
        out << ",\n";
        out << "      \"user_completion_throughput_GBps\": "
            << user_completion_throughput_GBps(result) << ",\n";
        out << "      \"makespan_throughput_GBps\": "
            << makespan_throughput_GBps(result) << ",\n";
        out << "      \"hbm_stats\": ";
        if (result.has_hbm) {
            const auto& s = result.hbm_stats;
            out << "{\"read_bytes\":" << s.read_bytes
                << ",\"write_bytes\":" << s.write_bytes
                << ",\"row_hits\":" << s.row_hits
                << ",\"row_misses\":" << s.row_misses
                << ",\"row_conflicts\":" << s.row_conflicts
                << ",\"row_hit_rate\":" << s.row_hit_rate()
                << ",\"refresh_count\":" << s.refresh_count
                << ",\"max_queue_occupancy\":" << s.max_queue_occupancy
                << ",\"bus_parallelism\":" << s.bus_parallelism()
                << ",\"pseudo_channels\":" << s.pseudo_channels
                << ",\"active_pseudo_channels\":" << s.active_pseudo_channels
                << ",\"max_pseudo_channel_accesses\":" << s.max_pseudo_channel_accesses
                << ",\"max_pseudo_channel_busy_ns\":" << s.max_pseudo_channel_busy_ns
                << ",\"avg_active_pseudo_channel_busy_ns\":"
                << s.avg_active_pseudo_channel_busy_ns
                << ",\"pseudo_channel_busy_skew\":"
                << s.pseudo_channel_busy_skew() << "},\n";
        } else {
            out << "null,\n";
        }
        out << "      \"hbf_stats\": ";
        if (result.has_hbf) {
            const auto& s = result.hbf_stats;
            out << "{\"read_requests\":" << s.read_requests
                << ",\"program_requests\":" << s.program_requests
                << ",\"erase_requests\":" << s.erase_requests
                << ",\"logical_read_bytes\":" << s.logical_read_bytes
                << ",\"physical_read_bytes\":" << s.physical_read_bytes
                << ",\"logical_write_bytes\":" << s.logical_write_bytes
                << ",\"physical_write_bytes\":" << s.physical_write_bytes
                << ",\"data_program_payload_bytes\":"
                << s.data_program_payload_bytes
                << ",\"mapping_program_payload_bytes\":"
                << s.mapping_program_payload_bytes
                << ",\"gc_relocation_payload_bytes\":"
                << s.gc_relocation_payload_bytes
                << ",\"page_reads\":" << s.page_reads
                << ",\"data_programs\":" << s.data_programs
                << ",\"page_programs\":" << s.page_programs
                << ",\"block_erases\":" << s.block_erases
                << ",\"mapping_entries\":" << s.mapping_entries
                << ",\"invalidations\":" << s.invalidations
                << ",\"total_pages\":" << s.total_pages
                << ",\"free_pages\":" << s.free_pages
                << ",\"valid_pages\":" << s.valid_pages
                << ",\"invalid_pages\":" << s.invalid_pages
                << ",\"pending_program_pages\":"
                << s.pending_program_pages
                << ",\"pending_mapping_publications\":"
                << s.pending_mapping_publications
                << ",\"static_unmaterialized_pages\":"
                << s.static_unmaterialized_pages
                << ",\"accounting_verified\":"
                << (s.accounting_verified ? "true" : "false")
                << ",\"static_reserved_pages\":" << s.static_reserved_pages
                << ",\"initial_logical_data_pages\":"
                << s.initial_logical_data_pages
                << ",\"initial_mapping_pages\":"
                << s.initial_mapping_pages
                << ",\"compact_initial_logical_data_pages\":"
                << s.compact_initial_logical_data_pages
                << ",\"compact_initial_mapping_pages\":"
                << s.compact_initial_mapping_pages
                << ",\"compact_live_logical_data_pages\":"
                << s.compact_live_logical_data_pages
                << ",\"compact_live_mapping_pages\":"
                << s.compact_live_mapping_pages
                << ",\"compact_retired_logical_data_pages\":"
                << s.compact_retired_logical_data_pages
                << ",\"compact_retired_mapping_pages\":"
                << s.compact_retired_mapping_pages
                << ",\"waf\":";
            if (const auto value = s.waf()) {
                out << *value;
            } else {
                out << "null";
            }
            out << ",\"waf_definition\":"
                << "\"physical_write_bytes/logical_write_bytes\""
                << ",\"resident_mapping_table_bytes\":"
                << s.resident_mapping_table_bytes
                << ",\"resident_mapping_table_bytes_per_stack\":"
                << s.resident_mapping_table_bytes_per_stack
                << ",\"resident_mapping_pages_per_stack\":"
                << s.resident_mapping_pages_per_stack
                << ",\"mapping_lookup_ops\":" << s.mapping_lookup_ops
                << ",\"mapping_user_lookup_ops\":"
                << s.mapping_user_lookup_ops
                << ",\"mapping_gc_lookup_ops\":"
                << s.mapping_gc_lookup_ops
                << ",\"mapping_update_ops\":" << s.mapping_update_ops
                << ",\"mapping_user_update_ops\":"
                << s.mapping_user_update_ops
                << ",\"mapping_gc_update_ops\":"
                << s.mapping_gc_update_ops
                << ",\"mapping_dram_wait_ops\":"
                << s.mapping_dram_wait_ops
                << ",\"mapping_dram_wait_work_ns\":"
                << s.mapping_dram_wait_ns
                << ",\"mapping_dram_wait_max_ns\":"
                << s.mapping_dram_wait_max_ns
                << ",\"mapping_dram_issue_busy_ns\":"
                << s.mapping_dram_issue_busy_ns
                << ",\"mapping_dram_resources\":"
                << s.mapping_dram_resources
                << ",\"mapping_page_programs\":" << s.mapping_page_programs
                << ",\"read_buffer_hits\":" << s.read_buffer_hits
                << ",\"read_buffer_misses\":" << s.read_buffer_misses
                << ",\"read_buffer_read_bytes\":" << s.read_buffer_read_bytes
                << ",\"write_buffer_hits\":" << s.write_buffer_hits
                << ",\"write_buffer_misses\":" << s.write_buffer_misses
                << ",\"write_buffer_flushes\":" << s.write_buffer_flushes
                << ",\"write_buffer_merged_bytes\":" << s.write_buffer_merged_bytes
                << ",\"write_buffer_read_hits\":" << s.write_buffer_read_hits
                << ",\"write_buffer_read_bytes\":" << s.write_buffer_read_bytes
                << ",\"write_buffer_slot_wait_ops\":" << s.write_buffer_slot_wait_ops
                << ",\"write_buffer_slot_wait_work_ns\":"
                << s.write_buffer_slot_wait_ns
                << ",\"read_splits\":" << s.read_splits
                << ",\"read_split_pages\":" << s.read_split_pages
                << ",\"page_read_admission_events\":"
                << s.page_read_admission_events
                << ",\"page_read_admission_waited_pages\":"
                << s.page_read_admission_waited_pages
                << ",\"page_read_admission_wait_work_ns\":"
                << s.page_read_admission_wait_ns
                << ",\"page_read_admission_max_wait_ns\":"
                << s.page_read_admission_max_wait_ns
                << ",\"flash_scheduler_enqueues\":" << s.flash_scheduler_enqueues
                << ",\"flash_scheduler_issues\":" << s.flash_scheduler_issues
                << ",\"gc_runs\":" << s.gc_runs
                << ",\"gc_relocations\":" << s.gc_relocations
                << ",\"gc_data_relocations\":" << s.gc_data_relocations
                << ",\"gc_mapping_relocations\":"
                << s.gc_mapping_relocations
                << ",\"gc_reclaimed_invalid_pages\":"
                << s.gc_reclaimed_invalid_pages
                << ",\"gc_user_blocked_runs\":" << s.gc_user_blocked_runs
                << ",\"ecc_decode_queue_wait_work_ns\":"
                << s.ecc_decode_queue_wait_ns
                << ",\"ecc_encode_queue_wait_work_ns\":"
                << s.ecc_encode_queue_wait_ns
                << ",\"ecc_decode_latency_work_ns\":"
                << s.ecc_decode_latency_work_ns
                << ",\"ecc_encode_latency_work_ns\":"
                << s.ecc_encode_latency_work_ns
                << ",\"ecc_decode_issue_busy_ns\":" << s.ecc_decode_issue_busy_ns
                << ",\"ecc_encode_issue_busy_ns\":" << s.ecc_encode_issue_busy_ns
                << ",\"ecc_decode_ops\":" << s.ecc_decode_ops
                << ",\"ecc_encode_ops\":" << s.ecc_encode_ops
                << ",\"ecc_codeword_bytes\":" << s.ecc_codeword_bytes
                << ",\"ecc_decode_codeword_bytes\":" << s.ecc_decode_codeword_bytes
                << ",\"ecc_encode_codeword_bytes\":" << s.ecc_encode_codeword_bytes
                << ",\"ecc_issue_parallelism\":" << s.ecc_issue_parallelism()
                << ",\"active_ecc_dies\":" << s.active_ecc_dies
                << ",\"ecc_max_inflight_per_die\":" << s.max_ecc_inflight_per_die
                << ",\"max_ecc_issue_busy_ns\":" << s.max_ecc_issue_busy_ns
                << ",\"avg_active_ecc_issue_busy_ns\":"
                << s.avg_active_ecc_issue_busy_ns
                << ",\"media_parallelism\":" << s.media_parallelism()
                << ",\"read_lane_parallelism\":" << s.read_lane_parallelism()
                << ",\"subarray_read_parallelism\":" << s.subarray_read_parallelism()
                << ",\"page_buffer_bank_parallelism\":"
                << s.page_buffer_bank_parallelism()
                << ",\"channel_parallelism\":" << s.channel_parallelism()
                << ",\"hbio_parallelism\":" << s.hbio_parallelism()
                << ",\"sequencer_parallelism\":" << s.sequencer_parallelism()
                << ",\"channels\":" << s.channels
                << ",\"active_channels\":" << s.active_channels
                << ",\"dies\":" << s.dies
                << ",\"active_dies\":" << s.active_dies
                << ",\"planes\":" << s.planes
                << ",\"active_planes\":" << s.active_planes
                << ",\"media_lanes\":" << s.media_lanes
                << ",\"active_media_lanes\":" << s.active_media_lanes
                << ",\"subarrays\":" << s.subarrays
                << ",\"active_subarrays\":" << s.active_subarrays
                << ",\"page_buffer_banks\":" << s.page_buffer_banks
                << ",\"active_page_buffer_banks\":"
                << s.active_page_buffer_banks
                << ",\"max_media_lane_busy_ns\":" << s.max_media_lane_busy_ns
                << ",\"avg_active_media_lane_busy_ns\":"
                << s.avg_active_media_lane_busy_ns
                << ",\"media_lane_skew\":" << s.media_lane_skew()
                << ",\"max_media_lane_reads\":" << s.max_media_lane_reads
                << ",\"avg_active_media_lane_reads\":"
                << s.avg_active_media_lane_reads
                << ",\"media_lane_read_skew\":" << s.media_lane_read_skew()
                << ",\"max_subarray_busy_ns\":" << s.max_subarray_busy_ns
                << ",\"avg_active_subarray_busy_ns\":"
                << s.avg_active_subarray_busy_ns
                << ",\"subarray_busy_skew\":" << s.subarray_busy_skew()
                << ",\"max_subarray_reads\":" << s.max_subarray_reads
                << ",\"avg_active_subarray_reads\":"
                << s.avg_active_subarray_reads
                << ",\"subarray_read_skew\":" << s.subarray_read_skew()
                << ",\"max_page_buffer_bank_busy_ns\":"
                << s.max_page_buffer_bank_busy_ns
                << ",\"avg_active_page_buffer_bank_busy_ns\":"
                << s.avg_active_page_buffer_bank_busy_ns
                << ",\"page_buffer_bank_skew\":"
                << s.page_buffer_bank_skew()
                << ",\"max_page_buffer_bank_reads\":"
                << s.max_page_buffer_bank_reads
                << ",\"avg_active_page_buffer_bank_reads\":"
                << s.avg_active_page_buffer_bank_reads
                << ",\"page_buffer_bank_read_skew\":"
                << s.page_buffer_bank_read_skew()
                << ",\"max_plane_media_busy_ns\":" << s.max_plane_media_busy_ns
                << ",\"avg_active_plane_media_busy_ns\":"
                << s.avg_active_plane_media_busy_ns
                << ",\"plane_media_skew\":" << s.plane_media_skew()
                << ",\"max_plane_ops\":" << s.max_plane_ops
                << ",\"avg_active_plane_ops\":" << s.avg_active_plane_ops
                << ",\"plane_op_skew\":" << s.plane_op_skew()
                << ",\"max_channel_busy_ns\":" << s.max_channel_busy_ns
                << ",\"avg_active_channel_busy_ns\":"
                << s.avg_active_channel_busy_ns
                << ",\"channel_busy_skew\":" << s.channel_busy_skew()
                << ",\"max_die_transactions\":" << s.max_die_transactions
                << ",\"avg_active_die_transactions\":"
                << s.avg_active_die_transactions
                << ",\"die_transaction_skew\":" << s.die_transaction_skew()
                << "},\n";
        } else {
            out << "null,\n";
        }
        out << "      \"external_backing_stats\": ";
        if (result.has_external_backing) {
            const auto& s = result.external_backing_stats;
            out << "{\"kind\":\""
                << hbfsim::physical::external::to_string(s.kind)
                << "\",\"read_requests\":" << s.read_requests
                << ",\"write_requests\":" << s.write_requests
                << ",\"read_bytes\":" << s.read_bytes
                << ",\"write_bytes\":" << s.write_bytes
                << ",\"media_channels\":" << s.media_channels
                << ",\"active_media_channels\":"
                << s.active_media_channels
                << ",\"max_device_outstanding\":"
                << s.max_device_outstanding
                << ",\"outstanding_wait_work_ns\":"
                << s.outstanding_wait_ns
                << ",\"controller_queue_wait_work_ns\":"
                << s.controller_queue_wait_ns
                << ",\"controller_issue_busy_ns\":"
                << s.controller_issue_busy_ns
                << ",\"controller_processing_work_ns\":"
                << s.controller_processing_work_ns
                << ",\"controller_active_span_ns\":"
                << s.controller_active_span_ns()
                << ",\"controller_utilization\":"
                << s.controller_utilization()
                << ",\"media_queue_wait_work_ns\":"
                << s.media_queue_wait_ns
                << ",\"media_read_latency_work_ns\":"
                << s.media_read_latency_work_ns
                << ",\"media_write_latency_work_ns\":"
                << s.media_write_latency_work_ns
                << ",\"media_read_busy_ns\":" << s.media_read_busy_ns
                << ",\"media_write_busy_ns\":" << s.media_write_busy_ns
                << ",\"media_active_span_ns\":"
                << s.media_active_span_ns()
                << ",\"media_utilization\":" << s.media_utilization()
                << ",\"m2s_payload_bytes\":" << s.m2s_payload_bytes
                << ",\"m2s_protocol_bytes\":" << s.m2s_protocol_bytes
                << ",\"m2s_wire_bytes\":" << s.m2s_wire_bytes
                << ",\"s2m_payload_bytes\":" << s.s2m_payload_bytes
                << ",\"s2m_protocol_bytes\":" << s.s2m_protocol_bytes
                << ",\"s2m_wire_bytes\":" << s.s2m_wire_bytes
                << ",\"m2s_queue_wait_work_ns\":"
                << s.m2s_queue_wait_ns
                << ",\"s2m_queue_wait_work_ns\":"
                << s.s2m_queue_wait_ns
                << ",\"m2s_busy_ns\":" << s.m2s_busy_ns
                << ",\"s2m_busy_ns\":" << s.s2m_busy_ns
                << ",\"transport_propagation_work_ns\":"
                << s.transport_propagation_work_ns
                << ",\"m2s_active_span_ns\":"
                << s.m2s_active_span_ns()
                << ",\"s2m_active_span_ns\":"
                << s.s2m_active_span_ns()
                << ",\"m2s_utilization\":"
                << s.m2s_utilization()
                << ",\"s2m_utilization\":"
                << s.s2m_utilization()
                << "},\n";
        } else {
            out << "null,\n";
        }
        if (!result.address_heatmap) {
            throw std::runtime_error(
                "scenario " + result.name +
                " completed without its required address heatmap snapshot");
        }
        out << "      \"address_heatmap\": ";
        hbfsim::physical::write_address_heatmap_json(
            out, *result.address_heatmap);
        out << ",\n";
        out << "      \"warnings\": [";
        for (std::size_t w = 0; w < result.warnings.size(); ++w) {
            if (w != 0) out << ", ";
            out << "\"" << json_escape(result.warnings[w]) << "\"";
        }
        out << "]\n";
        out << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    finish_output(out, output_path, "summary JSON");
}

void write_chrome_trace(
    const std::string& output_path,
    const std::vector<ScenarioResult>& results) {
    ensure_parent_dir(output_path);
    std::map<std::string, int> tids;
    auto tid_for = [&tids](const std::string& entity) -> int {
        const auto key = entity.empty() ? std::string{"unknown"} : entity;
        const auto found = tids.find(key);
        if (found != tids.end()) {
            return found->second;
        }
        const int next = static_cast<int>(tids.size() + 1);
        tids[key] = next;
        return next;
    };

    for (const auto& result : results) {
        for (const auto& row : result.completions) {
            for (const auto& span : row.spans) {
                (void)tid_for(result.name + "/" + span.entity);
            }
        }
    }

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot open Chrome trace output: " + output_path);
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n  \"traceEvents\": [\n";
    bool first = true;
    auto comma = [&]() {
        if (!first) {
            out << ",\n";
        }
        first = false;
    };

    for (const auto& [entity, tid] : tids) {
        comma();
        out << "    {\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << tid
            << ",\"args\":{\"name\":\"" << json_escape(entity) << "\"}}";
    }
    for (const auto& result : results) {
        for (const auto& row : result.completions) {
            for (const auto& span : row.spans) {
                comma();
                const auto entity = result.name + "/" + (span.entity.empty() ? row.resource_path : span.entity);
                out << "    {\"name\":\"" << json_escape(span.name)
                    << "\",\"cat\":\"" << json_escape(result.name + "/" + span.category)
                    << "\",\"ph\":\"X\",\"ts\":" << (span.start_ns / 1000.0)
                    << ",\"dur\":" << (span.duration_ns() / 1000.0)
                    << ",\"pid\":1,\"tid\":" << tid_for(entity)
                    << ",\"args\":{"
                    << "\"request\":\"" << json_escape(row.id) << "\","
                    << "\"tier\":\"" << hbfsim::physical::to_string(row.tier) << "\","
                    << "\"op\":\"" << hbfsim::physical::to_string(row.op) << "\","
                    << "\"entity\":\"" << json_escape(entity) << "\","
                    << "\"detail\":\"" << json_escape(span.detail) << "\""
                    << "}}";
            }
        }
    }
    out << "\n  ]\n}\n";
    finish_output(out, output_path, "Chrome trace");
}

TraceMode parse_trace_mode_value(const std::string& value) {
    if (value == "off") return TraceMode::Off;
    if (value == "summary") return TraceMode::Summary;
    if (value == "sampled") return TraceMode::Sampled;
    if (value == "full") return TraceMode::Full;
    throw std::runtime_error("unknown trace mode: " + value);
}

void usage(const char* argv0) {
    std::cerr << "usage:\n"
              << "  " << argv0 << " --version\n"
              << "  " << argv0 << " --config configs/scenario_compare/server-4k-hbf4x.cfg --trace path\n"
              << "      [--summary-csv path.csv] [--summary-json path.json] [--config-out path.cfg]\n"
              << "      [--chrome-trace path.json] [--max-ops N]\n"
              << "      [--trace-census-only true] (validate/count spans without device expansion)\n"
              << "\n"
              << "Advanced overrides:\n"
              << "  " << argv0 << " [--config path] --trace path [--line-size N] [--interarrival-ns N]\n"
              << "      [--expected-trace-sha256 HEX --expected-trace-bytes N]\n"
              << "      [--initial-image-trace path] (population only; stabilizes prefix placement)\n"
              << "  " << argv0 << " [--config path] --synthetic-sequential-read-bytes N"
              << " [--synthetic-sequential-read-base N]\n"
              << "      (lazy fixed-size reads; N/base must be line-size aligned; select a direct logical scenario)\n"
              << "      (config files apply first in argv order; any CLI option overrides them)\n"
              << "      [--hbm-capacity-bytes N] [--flat-hbm-bytes N]\n"
              << "      [--hbf-capacity-bytes N | --hbf-capacity-ratio N]\n"
              << "      [--layer-buffer-bytes N] (upper bound for one layer-streaming buffer)\n"
              << "      [--explicit-residency-contract true] with every residency-* field:\n"
              << "        unique-footprint, immutable-weight bytes/pages, runtime-overhead,\n"
              << "        block-table, active-buffer, KV-region/stride/logical-blocks/hot-blocks;\n"
              << "        requires the full digest-bound trace and rejects --max-ops\n"
              << "      [--behavioral-hbm-bytes N] (0 = complete configured HBM)\n"
              << "      [--behavioral-promotion-threshold N]"
                 " [--behavioral-history-pages N]\n"
              << "      [--max-outstanding-requests W] (per-tier page credits; 0 = open loop)\n"
              << "      [--max-hbm-outstanding-requests W] [--max-hbf-outstanding-requests W]\n"
              << "        (independent read-only direct-composition credit pools; mutually exclusive with shared W)\n"
              << "      [--address-heatmap-bins N] (1..8192; default 1024)\n"
              << "      [--base-die-link-read-bw GBps] [--base-die-link-write-bw GBps]\n"
              << "      [--base-die-link-latency-ns N]\n"
              << "      [--max-ops N]\n"
              << "      [--hbm-stacks N] [--hbm-channels N] [--hbm-pseudo-channels N]\n"
              << "      [--hbm-refresh bool] [--hbm-same-bank-refresh bool]\n"
              << "      [--hbm-trefi-ns N] [--hbm-trfc-ns N] [--hbm-trfcsb-ns N]\n"
              << "      [--hbm-queue-depth N] [--hbm-frfcfs-cap-ns N]\n"
              << "      [--hbm-bank-groups-per-pseudo-channel N] [--hbm-banks-per-group N]\n"
              << "      [--hbm-channel-row-size-bytes N] [--hbm-channel-width-bits N]\n"
              << "      [--hbm-burst-length N] [--hbm-pin-rate-gbps N]\n"
              << "      [--hbm-data-rate-per-command-clock N]\n"
              << "      [--hbm-tccd-s-cycles N] [--hbm-tccd-l-cycles N]\n"
              << "        HBM channel/pseudo-channel bandwidth, burst bytes, tCK and tBL are derived.\n"
              << "      [--hbm-address-mapping-ns N] [--hbm-trcdrd-ns N] [--hbm-trcdwr-ns N]\n"
              << "      [--hbm-tcl-ns N] [--hbm-tcwl-ns N] [--hbm-trp-ns N]\n"
              << "      [--hbm-tras-ns N] [--hbm-trc-ns N] [--hbm-twr-ns N]\n"
              << "      [--hbm-trtp-ns N] [--hbm-trrd-s-ns N] [--hbm-trrd-l-ns N]\n"
              << "      [--hbm-tfaw-ns N] [--hbm-twtr-s-ns N] [--hbm-twtr-l-ns N]\n"
              << "      [--hbm-trtw-ns N]\n"
              << "      [--hbf-stacks N] [--hbf-channels N] [--hbf-dies-per-channel N]\n"
              << "      [--hbf-planes-per-die N] [--hbf-blocks-per-plane N]\n"
              << "      [--hbf-pages-per-block N] [--hbf-page-size N] [--hbf-oob-bytes N]\n"
              << "      [--hbf-media-lanes-per-plane N] [--hbf-subarrays-per-plane N]\n"
              << "      [--hbf-page-buffer-banks-per-plane N]\n"
              << "      [--hbf-page-read-queue-depth-per-stack N]\n"
              << "      [--hbf-read-ns N] [--hbf-program-ns N]\n"
              << "      [--hbf-program-verify-ns N] [--hbf-erase-ns N]\n"
              << "      [--hbf-ecc-decode-latency-ns N] [--hbf-ecc-encode-latency-ns N]\n"
              << "      [--hbf-ecc-decode-raw-bw GBps-per-die]"
              << " [--hbf-ecc-encode-raw-bw GBps-per-die]\n"
              << "        ECC raw bandwidth includes page+OOB codeword bytes;"
              << " decode/encode share one issue port per die; latency >= II.\n"
              << "      [--hbf-channel-bw GBps] [--hbf-hbio-bw GBps] [--hbf-tsv-bw GBps]\n"
              << "      [--hbf-media-lane-bw GBps] [--hbf-logic-sram-bw GBps]\n"
              << "      [--hbf-page-buffer-bw GBps]\n"
              << "        channel/media-lane/page-buffer use raw page+OOB rates; TSV"
              << " shares commands and raw codewords; HBIO/logic-SRAM use decoded"
              << " payload GB/s.\n"
              << "      [--hbf-ctrl-dram-bytes N]"
                 " [--hbf-ctrl-dram-latency-ns N]"
                 " [--hbf-ctrl-dram-issue-ns N]\n"
              << "      [--hbf-flash-tsu-issue-ns N]\n"
              << "      [--hbf-logic-scheduler-issue-ns N]\n"
              << "      [--hbf-gc-low-watermark-pages N]\n"
              << "      [--hbf-gc-hard-watermark-pages N]\n"
              << "      [--hbf-gc-reserved-free-blocks-per-plane N]\n"
              << "      [--hbf-gc-wear-leveling-weight N]\n"
              << "      [--hbf-write-coalescing true|false]\n"
              << "      [--hbf-write-buffer-completion-requires-flush true|false]\n"
              << "      [--hbf-write-buffer-pages N]\n"
              << "      [--hbf-write-buffer-flush-threshold-pages N]\n"
              << "      [--hbf-hbm-write-buffer-bytes N] (0 = off; cooperative write staging)\n"
              << "      [--hbf-hbm-write-buffer-destage deferred|streamed]\n"
              << "      [--chrome-trace path.json] [--trace-mode off|summary|sampled|full]\n"
              << "      [--summary-csv path.csv] [--summary-json path.json] [--config-out path.cfg]\n"
              << "  " << argv0 << " --generate-llm path [--llm-tokens N] [--llm-layers N]\n"
              << "      [--llm-weight-base N] [--llm-kv-base N] [--llm-scratch-base N]\n"
              << "  " << argv0 << " --generate-semantic-llm path [--llm-tokens N] [--llm-layers N]\n"
              << "      [--llm-weight-base N] [--llm-kv-base N] [--llm-scratch-base N]\n"
              << "\n"
              << "Trace input is Ramulator-compatible text: <addr> <R|W>, one memory op per line.\n"
              << "Optional semantic form: <addr> <R|W> [bytes] "
                 "[model_weights|shared_context|generated_context|scratch|metadata] "
                 "[phase=N] [layer=N] [compute_ns=N].\n"
              << "Config files are key=value with the same names as long CLI options without '--'.\n"
              << "Prefer config files for stable HBM/HBF hardware, timing, flash geometry, and policy settings.\n"
              << "Keep run-local choices such as --trace, output paths, --chrome-trace, and --max-ops on the CLI.\n"
              << "Multiple --config files are applied in order; later CLI options override earlier config values.\n";
}

Options parse_args(int argc, char** argv) {
    Options options;
    // Two passes: config files apply first (in argv order), then every other
    // CLI option (in argv order) — so a CLI option always overrides config
    // values regardless of where --config appears on the command line.
    std::vector<std::pair<std::string, std::string>> cli_options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--version") {
            std::cout << "HBFSim scenario_compare " << HBFSIM_VERSION
                      << " (git " << HBFSIM_GIT_COMMIT
                      << (HBFSIM_GIT_DIRTY ? ", dirty-at-configure" : "")
                      << ")\n";
            std::exit(0);
        }
        if (!arg.starts_with("--")) {
            throw std::runtime_error("unknown argument: " + arg);
        }

        const auto value = need_value(arg.c_str());
        if (arg == "--config") {
            load_config_file(options, value);
        } else {
            cli_options.emplace_back(arg.substr(2), value);
        }
    }
    for (const auto& [key, value] : cli_options) {
        apply_option(options, key, value);
    }
    if (options.line_size == 0) {
        throw std::runtime_error("--line-size must be positive");
    }
    if (options.trace_path && options.synthetic_sequential_read_bytes) {
        throw std::runtime_error(
            "--trace and --synthetic-sequential-read-bytes are mutually exclusive");
    }
    if (options.expected_trace_sha256.has_value() !=
        options.expected_trace_file_bytes.has_value()) {
        throw std::runtime_error(
            "--expected-trace-sha256 and --expected-trace-bytes must be "
            "provided together");
    }
    if (options.expected_trace_sha256) {
        if (!options.trace_path) {
            throw std::runtime_error(
                "expected trace identity requires --trace");
        }
        if (!is_lower_hex_sha256(*options.expected_trace_sha256)) {
            throw std::runtime_error(
                "--expected-trace-sha256 must be 64 lowercase hex digits");
        }
    }
    if (options.initial_image_trace_path && !options.trace_path) {
        throw std::runtime_error(
            "--initial-image-trace requires an executed --trace workload");
    }
    if (options.synthetic_sequential_read_bytes) {
        if (*options.synthetic_sequential_read_bytes == 0 ||
            *options.synthetic_sequential_read_bytes % options.line_size != 0 ||
            options.synthetic_sequential_read_base % options.line_size != 0) {
            throw std::runtime_error(
                "synthetic sequential byte count/base must be positive and "
                "--line-size aligned");
        }
        if (options.max_ops != 0) {
            throw std::runtime_error(
                "--max-ops is not used with --synthetic-sequential-read-bytes; "
                "set the exact byte count instead");
        }
    } else if (options.synthetic_sequential_read_base != 0) {
        throw std::runtime_error(
            "--synthetic-sequential-read-base requires "
            "--synthetic-sequential-read-bytes");
    }
    if (options.generate_llm_path && options.generate_semantic_llm_path) {
        throw std::runtime_error("--generate-llm and --generate-semantic-llm are mutually exclusive");
    }
    if (options.synthetic_sequential_read_bytes &&
        (options.generate_llm_path || options.generate_semantic_llm_path)) {
        throw std::runtime_error(
            "synthetic sequential input cannot be combined with an LLM trace generator");
    }
    if (options.interarrival_ns < 0.0 || !std::isfinite(options.interarrival_ns)) {
        throw std::runtime_error("--interarrival-ns must be non-negative and finite");
    }
    if (options.address_heatmap_bins == 0 ||
        options.address_heatmap_bins > hbfsim::physical::kMaxAddressHeatmapBins) {
        throw std::runtime_error("--address-heatmap-bins must be in [1, 8192]");
    }
    const bool any_layer_streaming =
        scenario_selected(options, kLayerStreamingScenario) ||
        scenario_selected(options, kExternalLayerStreamingScenario);
    if (any_layer_streaming) {
        if (options.layer_buffer_bytes == 0) {
            throw std::runtime_error("--layer-buffer-bytes must be positive");
        }
    }
    const std::array<bool, 12> residency_fields{{
        options.residency_page_size_bytes.has_value(),
        options.residency_unique_footprint_bytes.has_value(),
        options.residency_immutable_weight_bytes.has_value(),
        options.residency_immutable_weight_pages.has_value(),
        options.residency_static_weight_pages.has_value(),
        options.residency_runtime_overhead_bytes.has_value(),
        options.residency_block_table_bytes.has_value(),
        options.residency_active_buffer_bytes.has_value(),
        options.residency_kv_region_begin.has_value(),
        options.residency_kv_block_stride_bytes.has_value(),
        options.residency_logical_kv_blocks.has_value(),
        options.residency_hot_kv_blocks.has_value(),
    }};
    const auto residency_field_count = static_cast<std::size_t>(std::count(
        residency_fields.begin(), residency_fields.end(), true));
    if (options.explicit_residency_contract) {
        if (!any_layer_streaming) {
            throw std::runtime_error(
                "--explicit-residency-contract requires a layer-streaming "
                "scenario");
        }
        if (!options.expected_trace_sha256) {
            throw std::runtime_error(
                "--explicit-residency-contract requires a digest-bound trace");
        }
        if (options.max_ops != 0) {
            throw std::runtime_error(
                "--explicit-residency-contract rejects --max-ops truncation");
        }
        if (residency_field_count != residency_fields.size()) {
            throw std::runtime_error(
                "--explicit-residency-contract requires every residency-* "
                "field");
        }
    } else if (residency_field_count != 0) {
        throw std::runtime_error(
            "residency-* fields require --explicit-residency-contract=true");
    }
    if (options.behavioral_history_pages == 0) {
        throw std::runtime_error(
            "--behavioral-history-pages must be positive");
    }
    if (options.behavioral_promotion_threshold < 2) {
        throw std::runtime_error(
            "--behavioral-promotion-threshold must be at least two");
    }
    if (options.behavioral_hbm_bytes > options.hbm_capacity_bytes) {
        throw std::runtime_error(
            "--behavioral-hbm-bytes cannot exceed --hbm-capacity-bytes");
    }
    if (options.hbf_hbm_write_buffer_bytes > options.hbm_capacity_bytes) {
        throw std::runtime_error(
            "--hbf-hbm-write-buffer-bytes cannot exceed --hbm-capacity-bytes");
    }
    auto require_positive_u64 = [](std::uint64_t value, const char* name) {
        if (value == 0) {
            throw std::runtime_error(std::string(name) + " must be positive");
        }
    };
    auto require_positive_f64 = [](double value, const char* name) {
        if (!(value > 0.0) || !std::isfinite(value)) {
            throw std::runtime_error(std::string(name) + " must be positive and finite");
        }
    };
    auto require_nonnegative_f64 = [](double value, const char* name) {
        if (value < 0.0 || !std::isfinite(value)) {
            throw std::runtime_error(
                std::string(name) + " must be non-negative and finite");
        }
    };
    require_positive_u64(options.hbm_capacity_bytes, "--hbm-capacity-bytes");
    require_positive_u64(options.hbm_stacks, "--hbm-stacks");
    require_positive_u64(options.hbm_channels, "--hbm-channels");
    require_positive_u64(options.hbm_pseudo_channels, "--hbm-pseudo-channels");
    require_positive_u64(
        options.hbm_bank_groups_per_pseudo_channel,
        "--hbm-bank-groups-per-pseudo-channel");
    require_positive_u64(options.hbm_banks_per_group, "--hbm-banks-per-group");
    require_positive_u64(
        options.hbm_channel_row_size_bytes, "--hbm-channel-row-size-bytes");
    require_positive_u64(options.hbm_channel_width_bits, "--hbm-channel-width-bits");
    require_positive_u64(options.hbm_burst_length, "--hbm-burst-length");
    require_positive_f64(options.hbm_pin_rate_Gbps, "--hbm-pin-rate-gbps");
    require_positive_u64(
        options.hbm_data_rate_per_command_clock,
        "--hbm-data-rate-per-command-clock");
    require_nonnegative_f64(
        options.hbm_address_mapping_ns, "--hbm-address-mapping-ns");
    require_positive_f64(options.hbm_trcdrd_ns, "--hbm-trcdrd-ns");
    require_positive_f64(options.hbm_trcdwr_ns, "--hbm-trcdwr-ns");
    require_positive_f64(options.hbm_tcl_ns, "--hbm-tcl-ns");
    require_positive_f64(options.hbm_tcwl_ns, "--hbm-tcwl-ns");
    require_positive_f64(options.hbm_trp_ns, "--hbm-trp-ns");
    require_positive_f64(options.hbm_tras_ns, "--hbm-tras-ns");
    require_positive_f64(options.hbm_trc_ns, "--hbm-trc-ns");
    require_positive_f64(options.hbm_twr_ns, "--hbm-twr-ns");
    require_positive_f64(options.hbm_trtp_ns, "--hbm-trtp-ns");
    require_positive_u64(options.hbm_tccd_s_cycles, "--hbm-tccd-s-cycles");
    require_positive_u64(options.hbm_tccd_l_cycles, "--hbm-tccd-l-cycles");
    require_positive_f64(options.hbm_trrd_s_ns, "--hbm-trrd-s-ns");
    require_positive_f64(options.hbm_trrd_l_ns, "--hbm-trrd-l-ns");
    require_positive_f64(options.hbm_tfaw_ns, "--hbm-tfaw-ns");
    require_positive_f64(options.hbm_twtr_s_ns, "--hbm-twtr-s-ns");
    require_positive_f64(options.hbm_twtr_l_ns, "--hbm-twtr-l-ns");
    require_positive_f64(options.hbm_trtw_ns, "--hbm-trtw-ns");
    require_positive_f64(options.hbm_trefi_ns, "--hbm-trefi-ns");
    require_positive_f64(options.hbm_trfc_ns, "--hbm-trfc-ns");
    require_positive_f64(options.hbm_trfcsb_ns, "--hbm-trfcsb-ns");
    require_positive_u64(options.hbm_queue_depth, "--hbm-queue-depth");
    require_nonnegative_f64(options.hbm_frfcfs_cap_ns, "--hbm-frfcfs-cap-ns");
    (void)HbmDevice(make_hbm_config(options));
    require_positive_f64(options.base_die_link_read_bw_GBps, "--base-die-link-read-bw");
    require_positive_f64(options.base_die_link_write_bw_GBps, "--base-die-link-write-bw");
    require_nonnegative_f64(
        options.base_die_link_latency_ns, "--base-die-link-latency-ns");
    require_positive_u64(options.hbf_stacks, "--hbf-stacks");
    require_positive_u64(options.hbf_channels, "--hbf-channels");
    require_positive_u64(options.hbf_dies_per_channel, "--hbf-dies-per-channel");
    require_positive_u64(options.hbf_planes_per_die, "--hbf-planes-per-die");
    require_positive_u64(options.hbf_pages_per_block, "--hbf-pages-per-block");
    require_positive_u64(options.hbf_page_size, "--hbf-page-size");
    require_positive_u64(options.hbf_media_lanes_per_plane, "--hbf-media-lanes-per-plane");
    require_positive_u64(
        options.hbf_page_buffer_banks_per_plane,
        "--hbf-page-buffer-banks-per-plane");
    if (options.hbf_pages_per_block > 1024) {
        throw std::runtime_error("--hbf-pages-per-block must be <= 1024");
    }
    if (options.hbf_page_size > 4096) {
        throw std::runtime_error("--hbf-page-size must be <= 4096 for first-stage SLC HBF");
    }
    if (options.hbf_oob_bytes >= options.hbf_page_size) {
        throw std::runtime_error("--hbf-oob-bytes must be smaller than --hbf-page-size");
    }
    apply_capacity_targets(options);
    require_positive_u64(options.hbf_blocks_per_plane, "--hbf-blocks-per-plane");
    if (options.flat_hbm_bytes > options.hbm_capacity_bytes) {
        throw std::runtime_error("--flat-hbm-bytes cannot exceed --hbm-capacity-bytes");
    }
    if (options.static_direct_hbm_bytes > options.hbm_capacity_bytes) {
        throw std::runtime_error(
            "--static-direct-hbm-bytes cannot exceed --hbm-capacity-bytes");
    }
    require_positive_f64(options.hbf_read_ns, "--hbf-read-ns");
    require_positive_f64(options.hbf_program_ns, "--hbf-program-ns");
    require_positive_f64(
        options.hbf_program_verify_ns, "--hbf-program-verify-ns");
    require_positive_f64(options.hbf_erase_ns, "--hbf-erase-ns");
    require_positive_f64(
        options.hbf_ecc_decode_latency_ns,
        "--hbf-ecc-decode-latency-ns");
    require_positive_f64(
        options.hbf_ecc_encode_latency_ns,
        "--hbf-ecc-encode-latency-ns");
    require_positive_f64(
        options.hbf_ecc_decode_raw_bw_GBps_per_die,
        "--hbf-ecc-decode-raw-bw");
    require_positive_f64(
        options.hbf_ecc_encode_raw_bw_GBps_per_die,
        "--hbf-ecc-encode-raw-bw");
    const auto ecc_codeword_bytes = options.hbf_page_size + options.hbf_oob_bytes;
    const double ecc_decode_ii_ns = hbfsim::physical::transfer_time_ns(
        ecc_codeword_bytes, options.hbf_ecc_decode_raw_bw_GBps_per_die);
    const double ecc_encode_ii_ns = hbfsim::physical::transfer_time_ns(
        ecc_codeword_bytes, options.hbf_ecc_encode_raw_bw_GBps_per_die);
    if (options.hbf_ecc_decode_latency_ns < ecc_decode_ii_ns) {
        throw std::runtime_error(
            "--hbf-ecc-decode-latency-ns must cover the full codeword initiation interval");
    }
    if (options.hbf_ecc_encode_latency_ns < ecc_encode_ii_ns) {
        throw std::runtime_error(
            "--hbf-ecc-encode-latency-ns must cover the full codeword initiation interval");
    }
    require_positive_f64(options.hbf_channel_bw_GBps, "--hbf-channel-bw");
    require_positive_f64(options.hbf_hbio_bw_GBps, "--hbf-hbio-bw");
    require_positive_f64(options.hbf_tsv_bw_GBps, "--hbf-tsv-bw");
    require_positive_f64(options.hbf_media_lane_bw_GBps, "--hbf-media-lane-bw");
    require_positive_f64(options.hbf_logic_sram_bw_GBps, "--hbf-logic-sram-bw");
    require_positive_f64(options.hbf_page_buffer_bw_GBps, "--hbf-page-buffer-bw");
    require_positive_f64(
        options.hbf_ctrl_dram_latency_ns, "--hbf-ctrl-dram-latency-ns");
    require_positive_f64(
        options.hbf_ctrl_dram_issue_ns, "--hbf-ctrl-dram-issue-ns");
    require_positive_f64(
        options.hbf_flash_tsu_issue_ns, "--hbf-flash-tsu-issue-ns");
    require_positive_f64(
        options.hbf_logic_scheduler_issue_ns, "--hbf-logic-scheduler-issue-ns");
    require_positive_u64(options.hbf_write_buffer_pages, "--hbf-write-buffer-pages");
    require_positive_u64(
        options.hbf_page_read_queue_depth_per_stack,
        "--hbf-page-read-queue-depth-per-stack");
    if (options.hbf_write_buffer_flush_threshold_pages >
        options.hbf_write_buffer_pages) {
        throw std::runtime_error(
            "--hbf-write-buffer-flush-threshold-pages cannot exceed "
            "--hbf-write-buffer-pages");
    }
    if (options.hbf_gc_reserved_free_blocks_per_plane >= options.hbf_blocks_per_plane) {
        throw std::runtime_error(
            "--hbf-gc-reserved-free-blocks-per-plane must be smaller than "
            "--hbf-blocks-per-plane");
    }
    require_nonnegative_f64(
        options.hbf_gc_wear_leveling_weight, "--hbf-gc-wear-leveling-weight");
    if ((options.generate_llm_path || options.generate_semantic_llm_path) &&
        (options.llm_tokens == 0 || options.llm_layers == 0)) {
        throw std::runtime_error("--llm-tokens and --llm-layers must be positive");
    }
    if (options.max_outstanding_requests != 0 &&
        (options.max_hbm_outstanding_requests != 0 ||
         options.max_hbf_outstanding_requests != 0)) {
        throw std::runtime_error(
            "--max-outstanding-requests cannot be combined with tier-specific "
            "outstanding windows");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        if (options.config_out_path) {
            write_resolved_config(*options.config_out_path, options);
            std::cout << "wrote resolved config: " << *options.config_out_path << '\n';
        }
        if (options.generate_llm_path || options.generate_semantic_llm_path) {
            generate_llm_trace(options);
            const auto& generated_path = options.generate_semantic_llm_path ?
                *options.generate_semantic_llm_path :
                *options.generate_llm_path;
            std::cout << "generated "
                      << (options.generate_semantic_llm_path ? "semantic " : "Ramulator-compatible ")
                      << "LLM trace: " << generated_path << '\n';
            if (!options.trace_path) {
                return 0;
            }
        }

        std::vector<TraceOp> ops;
        std::vector<TraceOp> initial_image_ops;
        std::optional<SequentialReadWorkload> sequential_workload;
        if (options.synthetic_sequential_read_bytes) {
            sequential_workload = make_synthetic_sequential_workload(options);
        } else {
            ops = load_ramulator_trace(options);
            if (options.initial_image_trace_path) {
                auto image_options = options;
                image_options.trace_path = options.initial_image_trace_path;
                image_options.max_ops = 0;
                initial_image_ops = load_ramulator_trace(image_options);
            }
        }
        RunProvenance provenance;
        provenance.command.reserve(static_cast<std::size_t>(argc));
        for (int i = 0; i < argc; ++i) {
            provenance.command.emplace_back(argv[i]);
        }
        provenance.working_directory = std::filesystem::current_path().string();
        std::error_code executable_error;
        auto executable_path = std::filesystem::absolute(argv[0], executable_error);
        if (!executable_error &&
            std::filesystem::is_regular_file(executable_path, executable_error) &&
            !executable_error) {
            executable_path = std::filesystem::weakly_canonical(
                executable_path,
                executable_error);
            if (!executable_error) {
                provenance.executable_path = executable_path.string();
                const auto executable_bytes =
                    std::filesystem::file_size(executable_path, executable_error);
                if (!executable_error &&
                    executable_bytes <= std::numeric_limits<std::uint64_t>::max()) {
                    provenance.executable_file_bytes =
                        static_cast<std::uint64_t>(executable_bytes);
                    provenance.executable_sha256 =
                        sha256_file(provenance.executable_path);
                }
            }
        }
        if (options.trace_path) {
            provenance.trace_sha256 = sha256_file(*options.trace_path);
            const auto trace_file_bytes =
                std::filesystem::file_size(*options.trace_path);
            if (trace_file_bytes > std::numeric_limits<std::uint64_t>::max()) {
                throw std::runtime_error("trace file size exceeds uint64_t range");
            }
            provenance.trace_file_bytes =
                static_cast<std::uint64_t>(trace_file_bytes);
            if (options.expected_trace_sha256 &&
                (provenance.trace_sha256 !=
                        *options.expected_trace_sha256 ||
                 provenance.trace_file_bytes !=
                        *options.expected_trace_file_bytes)) {
                throw std::runtime_error(
                    "executed trace identity differs from the digest-bound "
                    "residency config");
            }
        } else {
            provenance.trace_sha256 = sha256_text(
                synthetic_sequential_descriptor(options, *sequential_workload));
        }
        if (options.initial_image_trace_path) {
            provenance.initial_image_trace_sha256 =
                sha256_file(*options.initial_image_trace_path);
            const auto image_trace_file_bytes = std::filesystem::file_size(
                *options.initial_image_trace_path);
            if (image_trace_file_bytes >
                std::numeric_limits<std::uint64_t>::max()) {
                throw std::runtime_error(
                    "initial-image trace file size exceeds uint64_t range");
            }
            provenance.initial_image_trace_file_bytes =
                static_cast<std::uint64_t>(image_trace_file_bytes);
        }
        const auto trace_footprint_bytes = sequential_workload ?
            *options.synthetic_sequential_read_bytes :
            trace_unique_line_bytes(ops, options.line_size);
        const auto op_count = sequential_workload ?
            sequential_workload->request_count : ops.size();
        if (options.trace_census_only) {
            std::uint64_t logical_bytes = 0;
            std::uint64_t read_bytes = 0;
            std::uint64_t write_bytes = 0;
            std::uint64_t reads = 0;
            std::uint64_t writes = 0;
            std::set<std::uint64_t> phases;
            if (sequential_workload) {
                logical_bytes = *options.synthetic_sequential_read_bytes;
                read_bytes = logical_bytes;
                reads = sequential_workload->request_count;
            } else {
                for (const auto& op : ops) {
                    logical_bytes = checked_add_u64(
                        logical_bytes, op.bytes, "trace census logical bytes");
                    if (op.op == Op::Read) {
                        ++reads;
                        read_bytes = checked_add_u64(
                            read_bytes, op.bytes, "trace census read bytes");
                    } else {
                        ++writes;
                        write_bytes = checked_add_u64(
                            write_bytes, op.bytes, "trace census write bytes");
                    }
                    if (op.phase) {
                        phases.insert(*op.phase);
                    }
                }
            }
            std::cout
                << "{\"schema\":{\"name\":\"hbfsim.trace_census\",\"version\":1},"
                << "\"operations\":" << op_count << ','
                << "\"reads\":" << reads << ','
                << "\"writes\":" << writes << ','
                << "\"logical_bytes\":" << logical_bytes << ','
                << "\"read_bytes\":" << read_bytes << ','
                << "\"write_bytes\":" << write_bytes << ','
                << "\"unique_line_bytes\":" << trace_footprint_bytes << ','
                << "\"line_size\":" << options.line_size << ','
                << "\"explicit_phases\":" << phases.size() << ','
                << "\"trace_sha256\":\""
                << json_escape(provenance.trace_sha256) << "\"}\n";
            return 0;
        }
        const auto requests = sequential_workload ?
            std::vector<MemoryRequest>{} : make_memory_requests(ops);
        const auto initial_image_requests =
            make_memory_requests(initial_image_ops);
        const auto* initial_image_request_ptr =
            initial_image_requests.empty() ?
            nullptr :
            &initial_image_requests;
        std::vector<ScenarioResult> results;
        // The scenario table IS the scenario set: name validation,
        // selection, and execution all walk this one list, in this order.
        // Adding a composition point is one entry here — never a new
        // branch per concern.
        struct ScenarioSpec {
            const char* name;
            std::function<ScenarioResult()> run;
        };
        const auto run_direct_source = [&] (
            const char* scenario_name,
            const DirectPolicy& policy) {
            if (sequential_workload) {
                return run_direct(
                    policy,
                    scenario_name,
                    *sequential_workload,
                    options);
            }
            return run_direct(
                policy,
                scenario_name,
                requests,
                options,
                initial_image_request_ptr);
        };
        const std::vector<ScenarioSpec> scenario_table = {
            {kAllHbmScenario,
             [&] {
                 return run_direct_source(
                     kAllHbmScenario, all_hbm_policy());
             }},
            {kAllHbfScenario,
             [&] {
                 return run_direct_source(
                     kAllHbfScenario, all_hbf_policy());
             }},
            {kFlatScenario,
             [&] { return run_direct_source(
                       kFlatScenario,
                       flat_policy(options.flat_hbm_bytes)); }},
            {kStaticDirectScenario,
             [&] { return run_direct_source(
                       kStaticDirectScenario,
                       read_only_direct_policy(options.static_direct_hbm_bytes)); }},
            {kDemandFillScenario,
             [&] { return run_behavioral_tiering(
                       requests,
                       options,
                       BehavioralAdmissionPolicy::AlwaysAdmit,
                       kDemandFillScenario,
                       initial_image_request_ptr); }},
            {kBehavioralTieringScenario,
             [&] { return run_behavioral_tiering(
                       requests,
                       options,
                       BehavioralAdmissionPolicy::ReuseFiltered,
                       kBehavioralTieringScenario,
                       initial_image_request_ptr); }},
            {kLayerStreamingScenario,
             [&] { return run_layer_streaming(
                       requests, options, BackingTier::Hbf); }},
            {kExternalLayerStreamingScenario,
             [&] { return run_layer_streaming(
                       requests, options, BackingTier::External); }},
        };
        for (const auto& wanted : options.scenarios) {
            const bool known = std::any_of(
                scenario_table.begin(), scenario_table.end(),
                [&wanted](const ScenarioSpec& spec) { return wanted == spec.name; });
            if (!known) {
                throw std::runtime_error("unknown scenario in --scenarios: " + wanted);
            }
        }
        if ((options.max_hbm_outstanding_requests != 0 ||
             options.max_hbf_outstanding_requests != 0) &&
            scenario_selected(options, kLayerStreamingScenario)) {
            throw std::runtime_error(
                "tier-specific outstanding windows are defined only for "
                "read-only direct compositions; exclude " +
                std::string(kLayerStreamingScenario));
        }
        if ((options.max_hbm_outstanding_requests != 0 ||
             options.max_hbf_outstanding_requests != 0) &&
            scenario_selected(options, kExternalLayerStreamingScenario)) {
            throw std::runtime_error(
                "tier-specific outstanding windows are defined only for "
                "read-only direct compositions; exclude " +
                std::string(kExternalLayerStreamingScenario));
        }
        if ((options.max_hbm_outstanding_requests != 0 ||
             options.max_hbf_outstanding_requests != 0) &&
            (scenario_selected(options, kDemandFillScenario) ||
             scenario_selected(options, kBehavioralTieringScenario))) {
            throw std::runtime_error(
                "tier-specific outstanding windows are defined only for "
                "read-only direct compositions; exclude the behavioral "
                "tiering scenarios");
        }
        if (sequential_workload &&
            (scenario_selected(options, kLayerStreamingScenario) ||
             scenario_selected(options, kExternalLayerStreamingScenario) ||
             scenario_selected(options, kDemandFillScenario) ||
             scenario_selected(options, kBehavioralTieringScenario))) {
            throw std::runtime_error(
                "--synthetic-sequential-read-bytes currently supports "
                "direct scenarios only; exclude behavioral and "
                "layer-streaming scenarios");
        }
        for (const auto& spec : scenario_table) {
            if (scenario_selected(options, spec.name)) {
                results.push_back(spec.run());
            }
        }

        double first_arrival_ns = 0.0;
        double last_arrival_ns = 0.0;
        if (sequential_workload) {
            first_arrival_ns = sequential_workload->first_arrival_ns;
            last_arrival_ns = sequential_workload->first_arrival_ns +
                static_cast<double>(sequential_workload->request_count - 1) *
                    sequential_workload->interarrival_ns;
        } else {
            const auto [first_arrival, last_arrival] = std::minmax_element(
                ops.begin(), ops.end(), [](const TraceOp& lhs, const TraceOp& rhs) {
                    return lhs.arrival_ns < rhs.arrival_ns;
                });
            first_arrival_ns = first_arrival->arrival_ns;
            last_arrival_ns = last_arrival->arrival_ns;
        }
        for (auto& result : results) {
            if (!result.owns_arrival_frontier) {
                result.first_arrival_ns = first_arrival_ns;
                result.last_arrival_ns = last_arrival_ns;
            }
        }

        const bool sanity_ok = run_sanity_checks(results, options, trace_footprint_bytes);
        for (auto& result : results) {
            finalize_latency_distributions(
                result,
                sequential_workload.has_value());
        }
        print_run_config(options, op_count, trace_footprint_bytes);
        print_result_table(results);
        for (const auto& result : results) {
            print_device_stats(result);
        }
        std::cout << "\nSANITY: " << (sanity_ok ? "PASS" : "FAIL") << '\n';

        if (options.summary_csv_path) {
            write_summary_csv(*options.summary_csv_path, results);
            std::cout << "wrote summary CSV: " << *options.summary_csv_path << '\n';
        }
        if (options.summary_json_path) {
            write_summary_json(
                *options.summary_json_path,
                options,
                provenance,
                op_count,
                trace_footprint_bytes,
                results,
                sanity_ok);
            std::cout << "wrote summary JSON: " << *options.summary_json_path << '\n';
        }
        if (options.chrome_trace_path) {
            write_chrome_trace(*options.chrome_trace_path, results);
            std::cout << "wrote Chrome trace: " << *options.chrome_trace_path << '\n';
        }
        return sanity_ok ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
